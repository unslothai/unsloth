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
import { applySidebarLayout, sidebarLayout, type SidebarLayout } from "../sidebar-layout";
import {
  SIDEBAR_ORGANIZATION_STORAGE_KEY,
  useSidebarOrganizationStore as store,
} from "../stores/sidebar-organization-store";

export function useSidebarSectionsSync(): void {
  useEffect(() => {
    const account = browserAccountKey();
    let mounted = true;
    let request: AbortController | undefined;
    let timer: ReturnType<typeof setTimeout>;

    function allowed() {
      return mounted && !accountTransitionPending() &&
        !localStorage.getItem(BROWSER_ACCOUNT_FENCE_KEY) && browserAccountKey() === account &&
        hasSettledAuthSession() && !isBackendDownForDesktopUpdate();
    }

    function schedule() {
      clearTimeout(timer);
      timer = setTimeout(refresh, 0);
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
        edit = { id: createNavigationNonce(), layout: snapshot };
        store.setState({ pendingLayoutEdit: edit });
      }
      const sent = edit?.layout ?? snapshot;
      const cached = localStorage.getItem(SIDEBAR_ORGANIZATION_STORAGE_KEY);
      const controller = new AbortController();
      request = controller;
      const timeout = setTimeout(() => controller.abort(), 15_000);
      try {
        const response = await authFetch("/api/chat/sidebar-sections", {
          signal: controller.signal,
          ...(edit && {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              operationId: edit.id,
              base: base ? JSON.parse(base) : null,
              layout: JSON.parse(sent),
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
          // Keep the new local view. Its next diff starts at the snapshot just acknowledged.
          if (edit) store.setState({ syncedLayout: sent, pendingLayoutEdit: undefined });
          schedule();
          return;
        }
        const update = applySidebarLayout(now, remote);
        const syncedLayout = JSON.stringify(sidebarLayout({ ...now, ...update }));
        if (edit || base !== syncedLayout || sent !== syncedLayout) {
          store.setState({ ...update, syncedLayout, pendingLayoutEdit: undefined });
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

    const unsubscribe = store.subscribe((state, previous) => {
      if (state.syncedLayout !== previous.syncedLayout) return;
      if (state.customSections === previous.customSections && state.manualOrder === previous.manualOrder &&
        state.sectionByChatId === previous.sectionByChatId && state.sectionByPageId === previous.sectionByPageId &&
        state.sectionByProjectId === previous.sectionByProjectId && state.sectionOrder === previous.sectionOrder) return;
      if (JSON.stringify(sidebarLayout(state)) !== state.syncedLayout) schedule();
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
      clearTimeout(timer);
      window.clearInterval(poll);
      window.removeEventListener("focus", visible);
      window.removeEventListener("online", visible);
      window.removeEventListener(AUTH_SESSION_STORED_EVENT, visible);
      document.removeEventListener("visibilitychange", visible);
    };
  }, []);
}
