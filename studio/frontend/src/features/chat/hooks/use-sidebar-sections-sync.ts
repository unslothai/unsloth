// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { toast } from "sonner";
import {
  AUTH_SESSION_STORED_EVENT,
  authFetch,
  ensureAuthSessionMark,
  getAuthSessionMark,
  hasSettledAuthSession,
} from "@/features/auth";
import {
  accountTransitionPending,
  BROWSER_ACCOUNT_FENCE_KEY,
  browserAccountKey,
} from "@/lib/account-transition";
import { isBackendDownForDesktopUpdate } from "@/lib/desktop-update-activity";
import { createNavigationNonce } from "@/lib/navigation-nonce";
import { applySidebarLayout, observeSidebarLayout, sidebarLayout, type SidebarLayout } from "../sidebar-layout";
import {
  SIDEBAR_ORGANIZATION_STORAGE_KEY,
  useSidebarOrganizationStore as store,
} from "../stores/sidebar-organization-store";

export function useSidebarSectionsSync(): void {
  useEffect(() => {
    const account = browserAccountKey();
    let mounted = true;
    let request: AbortController | undefined;
    let scheduled = false;
    let mirrored = false;

    function allowed() {
      return mounted && !accountTransitionPending() &&
        !localStorage.getItem(BROWSER_ACCOUNT_FENCE_KEY) && browserAccountKey() === account &&
        hasSettledAuthSession() && !isBackendDownForDesktopUpdate();
    }

    function schedule() {
      if (scheduled) return;
      scheduled = true;
      queueMicrotask(() => {
        scheduled = false;
        void refresh();
      });
    }

    async function refresh() {
      if (request || !allowed()) return;
      const session = ensureAuthSessionMark();
      const current = () => allowed() && session === getAuthSessionMark();
      const state = store.getState();
      const base = state.syncedLayout;
      const layout = sidebarLayout(state);
      const snapshot = JSON.stringify(layout);
      const dirty = base === undefined
        ? state.customSections.length > 0 || layout.sectionOrder[0] === "projects"
        : base !== snapshot;

      // Freeze the edit before sending it. Reloads and other tabs can retry the same operation
      // after a lost reply, even when the user has already made further changes.
      let edit = state.pendingLayoutEdit;
      if (!edit && dirty) {
        if (mirrored) return;
        edit = { id: createNavigationNonce(), layout: snapshot, orderMoves: state.orderMoves };
      }
      const sent = edit?.layout ?? snapshot;
      const controller = new AbortController();
      request = controller;
      const timeout = setTimeout(() => controller.abort(), 15_000);
      try {
        if (edit !== state.pendingLayoutEdit || (!edit && state.orderMoves?.length)) {
          store.setState({ pendingLayoutEdit: edit, orderMoves: undefined });
        }
        const cached = localStorage.getItem(SIDEBAR_ORGANIZATION_STORAGE_KEY);
        const response = await authFetch("/api/chat/sidebar-sections", {
          signal: controller.signal,
          ...(edit && {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              operationId: edit.id,
              base: base ? JSON.parse(base) : null,
              layout: JSON.parse(sent),
              orderMoves: edit.orderMoves,
            }),
          }),
        }, { beforeRetry: () => {
          if (!current()) throw new Error("Sidebar session changed");
        } });
        if (!response.ok) throw new Error(String(response.status));
        const remote: SidebarLayout = await response.json();
        if (!current()) return;

        // A peer's storage event may still be queued when the response arrives.
        if (cached !== localStorage.getItem(SIDEBAR_ORGANIZATION_STORAGE_KEY)) {
          store.persist.rehydrate();
        }
        const now = store.getState();
        if (now.syncedLayout !== base || now.pendingLayoutEdit?.id !== edit?.id) {
          schedule();
          return;
        }
        if (JSON.stringify(sidebarLayout(now)) !== sent) {
          if (edit) {
            // The tab that acknowledges the pending operation can send the next edit.
            mirrored = false;
            store.setState({ syncedLayout: sent, pendingLayoutEdit: undefined });
          } else if (!mirrored && base === undefined) {
            // The first GET started before this local edit. Use the pre-edit view
            // as the base for the user's next write.
            store.setState({ syncedLayout: sent });
          }
          schedule();
          return;
        }
        const update = applySidebarLayout(now, remote);
        const syncedLayout = JSON.stringify(sidebarLayout({ ...now, ...update }));
        if (edit || base !== syncedLayout || sent !== syncedLayout || now.orderMoves?.length) {
          store.setState({
            ...update,
            syncedLayout,
            pendingLayoutEdit: undefined,
            orderMoves: undefined,
          });
        }
      } catch {
        if (edit && current()) {
          toast.error("Could not sync sidebar sections. Unsloth will retry when connected.", {
            id: "sidebar-sections-sync",
          });
        }
      } finally {
        clearTimeout(timeout);
        request = undefined;
      }
    }

    const unsubscribe = observeSidebarLayout((fromStorage) => {
      mirrored = fromStorage;
      if (!fromStorage) schedule();
    });
    const visible = () => { if (document.visibilityState === "visible") void refresh(); };
    const poll = window.setInterval(visible, 30_000);
    window.addEventListener("focus", visible);
    window.addEventListener("online", visible);
    window.addEventListener(AUTH_SESSION_STORED_EVENT, visible);
    document.addEventListener("visibilitychange", visible);
    visible();
    return () => {
      mounted = false;
      request?.abort();
      unsubscribe();
      window.clearInterval(poll);
      window.removeEventListener("focus", visible);
      window.removeEventListener("online", visible);
      window.removeEventListener(AUTH_SESSION_STORED_EVENT, visible);
      document.removeEventListener("visibilitychange", visible);
    };
  }, []);
}
