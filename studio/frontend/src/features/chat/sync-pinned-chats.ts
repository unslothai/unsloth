// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";
import { AUTH_SESSION_MARK_KEY, hasAuthToken } from "@/features/auth";
import { getApiBase, isTauri } from "@/lib/api-base";
import { createNavigationNonce } from "@/lib/navigation-nonce";
import { createChatPinsApi } from "./api/chat-pins-api";
import {
  discardUnboundPinEdits,
  setPinnedChatsActions,
  usePinnedChatsStore,
} from "./stores/pinned-chats-store";
import {
  preserveFallbackPins,
  readLegacyPinnedChatIds,
} from "./utils/legacy-pinned-chats";
import { createPinnedChatsJournal } from "./utils/pinned-chats-journal";
import { createPinnedChatsSync } from "./utils/pinned-chats-sync";
import { createPinnedChatsTransport } from "./utils/pinned-chats-transport";
import { isRetryablePinError } from "./utils/settings-retry";
import { onChatThreadsDeleted } from "./utils/chat-thread-tombstones";

function scope(): string {
  // Desktop's backend port changes between launches of the same installation.
  const server = isTauri
    ? window.location.origin
    : getApiBase() || window.location.origin;
  let mark = localStorage.getItem(AUTH_SESSION_MARK_KEY);
  if (!mark) {
    // Pre-mark sessions get one on token refresh, stranding the journal: set it now.
    mark = createNavigationNonce();
    localStorage.setItem(AUTH_SESSION_MARK_KEY, mark);
  }
  return `${server}|${mark}`;
}

function createRuntime(sessionScope: string) {
  const controller = new AbortController();
  const journal = createPinnedChatsJournal(localStorage, sessionScope);
  const legacy = readLegacyPinnedChatIds(localStorage);
  // Re-sent every refresh: legacy import may create rows later; tombstones keep unpins.
  const readLegacy = () => legacy;
  const api = createChatPinsApi(controller.signal);
  const transport = createPinnedChatsTransport({
    ...api,
    migrateLegacy: (ids) =>
      api.migrateLegacy([...new Set([...ids, ...readLegacy()])]),
    readLegacy,
    readFallback: journal.readFallback,
    writeFallback: journal.writeFallback,
    onUnsupported: () =>
      toast.info(
        "This server saves pins in this browser. They will sync after the server is updated.",
        { id: "pinned-chats-compatibility" },
      ),
  });
  const sync = createPinnedChatsSync({
    ...transport,
    readLegacy,
    readPending: journal.readPending,
    writePending: journal.writePending,
    isRetryable: isRetryablePinError,
    publish: (pinnedIds) => usePinnedChatsStore.setState({ pinnedIds }),
    onError: () =>
      toast.error("Could not sync pinned chats. Try again when connected.", {
        id: "pinned-chats-sync-error",
      }),
  });
  // Every delete path unpins what it removed, so a never-imported comparison's pin cannot block.
  const stopWatchingDeletes = onChatThreadsDeleted((threadIds, pairIds) => {
    const pinned = new Set(usePinnedChatsStore.getState().pinnedIds);
    const removed = [...threadIds, ...pairIds].filter((id) => pinned.has(id));
    if (removed.length > 0) void sync.setPinned(removed, false).catch(() => undefined);
  });
  setPinnedChatsActions({
    setPinned: (ids, pinned) => {
      void sync.setPinned(ids, pinned).catch(() => undefined);
    },
  });
  return {
    scope: sessionScope,
    refresh: sync.refresh,
    reset(clearPending: boolean) {
      controller.abort();
      stopWatchingDeletes();
      transport.reset();
      sync.reset();
      setPinnedChatsActions(null);
      if (clearPending) {
        preserveFallbackPins(localStorage, journal.readFallback);
        journal.clear();
      }
    },
  };
}

let runtime: ReturnType<typeof createRuntime> | undefined;

export async function refreshPinnedChats(): Promise<void> {
  if (!hasAuthToken()) return;
  const sessionScope = scope();
  if (runtime?.scope !== sessionScope) {
    runtime?.reset(true);
    runtime = createRuntime(sessionScope);
  }
  await runtime.refresh();
}

/** Unmount retains durable work; sign-out/account changes explicitly discard it. */
export function resetPinnedChats({
  clearPending = false,
}: { clearPending?: boolean } = {}): void {
  const previous = runtime;
  runtime = undefined;
  previous?.reset(clearPending);
  if (clearPending) discardUnboundPinEdits();
}
