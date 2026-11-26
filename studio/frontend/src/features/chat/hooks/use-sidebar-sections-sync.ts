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
  BROWSER_ACCOUNT_KEY,
  sameBrowserAccount,
} from "@/lib/account-transition";
import { isBackendDownForDesktopUpdate } from "@/lib/desktop-update-activity";
import { createNavigationNonce } from "@/lib/navigation-nonce";
import { applySidebarLayout, compactSidebarEdit, observeSidebarLayout, sidebarLayout, type SidebarLayout } from "../sidebar-layout";
import {
  reconcilePersistedSidebarOrganization,
  useSidebarOrganizationStore as store,
} from "../stores/sidebar-organization-store";

const SIDEBAR_POLL_INTERVAL_MS = 30_000;

export function useSidebarSectionsSync(): void {
  useEffect(() => {
    let account = localStorage.getItem(BROWSER_ACCOUNT_KEY);
    let mounted = true;
    let request: AbortController | undefined;
    let scheduled = false;
    let peerEditedAt: number | undefined;

    function allowed() {
      if (!mounted || accountTransitionPending() ||
        localStorage.getItem(BROWSER_ACCOUNT_FENCE_KEY) || !sameBrowserAccount(account) ||
        !hasSettledAuthSession() || isBackendDownForDesktopUpdate()) return false;
      // Keep the upgraded marker so another account cannot match by username alone.
      account = localStorage.getItem(BROWSER_ACCOUNT_KEY);
      return true;
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
      // The storage event for a peer's newer edit may still be queued.
      reconcilePersistedSidebarOrganization();
      const state = store.getState();
      const base = state.syncedLayout;
      const layout = sidebarLayout(state);
      const snapshot = JSON.stringify(layout);
      const dirty = state.orderMoves?.some((move) => move.afterUnpin) || (base === undefined
        ? state.customSections.length > 0 || layout.sectionOrder[0] === "projects"
        : base !== snapshot);

      // Freeze the edit before sending it. Reloads and other tabs can retry the same operation
      // after a lost reply, even when the user has already made further changes.
      let edit = state.pendingLayoutEdit;
      if (!edit && dirty) {
        // Give the editing tab a poll interval to save before a peer takes over.
        if (peerEditedAt !== undefined && Date.now() - peerEditedAt < SIDEBAR_POLL_INTERVAL_MS) return;
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
        const fetchLayout = (body?: string) => authFetch("/api/chat/sidebar-sections", {
          signal: controller.signal,
          ...(body && {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body,
          }),
        }, { beforeRetry: () => {
          if (!current()) throw new Error("Sidebar session changed");
        } });
        const payload = edit && {
          operationId: edit.id,
          base: base ? (JSON.parse(base) as SidebarLayout) : null,
          layout: JSON.parse(sent) as SidebarLayout,
          orderMoves: edit.orderMoves,
        };
        const body = payload && JSON.stringify(payload);
        let response = await fetchLayout(body);
        if (response.status === 413 && body && payload?.base && current()) {
          const compact = compactSidebarEdit(payload.base, payload.layout, payload.orderMoves);
          const smaller = JSON.stringify({ ...payload, ...compact });
          if (smaller.length < body.length) response = await fetchLayout(smaller);
        }
        if (!response.ok) {
          if ((response.status === 413 || response.status === 422) && edit && current()) {
            reconcilePersistedSidebarOrganization();
            const now = store.getState();
            if (now.pendingLayoutEdit?.id === edit.id) {
              // The server refused the body before applying it. Keep the visible edits retryable.
              const latest = sidebarLayout(now);
              const orderMoves = [...(edit.orderMoves ?? []), ...(now.orderMoves ?? [])].filter(
                ({ id, targetId, sectionId }) => {
                  const order = sectionId === null ? latest.sectionOrder : latest.rowOrder[sectionId];
                  return order?.includes(id) && order.includes(targetId);
                },
              );
              store.setState({
                pendingLayoutEdit: undefined,
                orderMoves,
              });
              if (JSON.stringify(latest) !== sent) schedule();
            }
          }
          throw new Error(String(response.status));
        }
        const remote: SidebarLayout = await response.json();
        if (!current()) return;

        reconcilePersistedSidebarOrganization();
        const now = store.getState();
        if (now.syncedLayout !== base || now.pendingLayoutEdit?.id !== edit?.id) {
          schedule();
          return;
        }
        if (JSON.stringify(sidebarLayout(now)) !== sent || now.orderMoves?.some((move) => move.afterUnpin)) {
          if (edit) {
            // The tab that acknowledges the pending operation can send the next edit.
            peerEditedAt = undefined;
            store.setState({ syncedLayout: sent, pendingLayoutEdit: undefined });
          } else if (peerEditedAt === undefined && base === undefined) {
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
      peerEditedAt = fromStorage ? Date.now() : undefined;
      if (!fromStorage) schedule();
    });
    const visible = () => { if (document.visibilityState === "visible") void refresh(); };
    const poll = window.setInterval(visible, SIDEBAR_POLL_INTERVAL_MS);
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
