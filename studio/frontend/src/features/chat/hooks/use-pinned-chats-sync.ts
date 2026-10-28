// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_STORED_EVENT,
  AUTH_SESSION_MARK_KEY,
  AUTH_TOKEN_KEY,
  hasAuthToken,
} from "@/features/auth";
import {
  PIN_JOURNAL_PREFIX,
  discardUnboundPinEdits,
} from "../stores/pinned-chats-store";

// Loaded on mount, not at startup: the sync runtime would push the eager bundle over budget.
let runtime: Promise<typeof import("../sync-pinned-chats")> | undefined;
const loadRuntime = () =>
  (runtime ??= import("../sync-pinned-chats").catch((error: unknown) => {
    // Browsers refetch a failed dynamic import, so the next focus/online/poll retries it.
    runtime = undefined;
    throw error;
  }));

/** Keep pins current across clients, and discard in-flight work at session boundaries. */
export function usePinnedChatsSync(): void {
  useEffect(() => {
    let sessionMark = localStorage.getItem(AUTH_SESSION_MARK_KEY);
    const refresh = () => {
      if (hasAuthToken() && document.visibilityState === "visible") {
        void loadRuntime()
          .then((sync) => {
            const refreshed = sync.refreshPinnedChats();
            // Its synchronous scope() may have just backfilled the mark of a pre-mark session.
            sessionMark = localStorage.getItem(AUTH_SESSION_MARK_KEY);
            return refreshed;
          })
          .catch(() => undefined);
        sessionMark = localStorage.getItem(AUTH_SESSION_MARK_KEY);
      }
    };
    const reconcileSession = () => {
      // Synchronous, so a runtime that fails to load cannot carry edits into the next session.
      discardUnboundPinEdits();
      void loadRuntime()
        .then((sync) => sync.resetPinnedChats({ clearPending: true }))
        .catch(() => undefined);
      sessionMark = localStorage.getItem(AUTH_SESSION_MARK_KEY);
      refresh();
    };
    const onStorage = (event: StorageEvent) => {
      if (event.key?.startsWith(PIN_JOURNAL_PREFIX)) refresh();
      if (
        event.key === null ||
        event.key === AUTH_SESSION_MARK_KEY ||
        event.key === AUTH_TOKEN_KEY
      ) {
        const nextMark = localStorage.getItem(AUTH_SESSION_MARK_KEY);
        if (nextMark !== sessionMark || !hasAuthToken()) reconcileSession();
      }
    };
    window.addEventListener(AUTH_SESSION_CLEARED_EVENT, reconcileSession);
    window.addEventListener(AUTH_SESSION_STORED_EVENT, reconcileSession);
    window.addEventListener("storage", onStorage);
    window.addEventListener("focus", refresh);
    window.addEventListener("online", refresh);
    document.addEventListener("visibilitychange", refresh);
    const timer = window.setInterval(refresh, 30_000);
    refresh();
    return () => {
      window.clearInterval(timer);
      window.removeEventListener(AUTH_SESSION_CLEARED_EVENT, reconcileSession);
      window.removeEventListener(AUTH_SESSION_STORED_EVENT, reconcileSession);
      window.removeEventListener("storage", onStorage);
      window.removeEventListener("focus", refresh);
      window.removeEventListener("online", refresh);
      document.removeEventListener("visibilitychange", refresh);
      const clearPending =
        !hasAuthToken() ||
        localStorage.getItem(AUTH_SESSION_MARK_KEY) !== sessionMark;
      if (clearPending) discardUnboundPinEdits();
      void loadRuntime()
        .then((sync) => sync.resetPinnedChats({ clearPending }))
        .catch(() => undefined);
    };
  }, []);
}
