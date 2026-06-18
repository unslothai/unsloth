// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { hasAuthToken } from "@/features/auth";
import { useEffect } from "react";
import { getExportStatus, streamExportLogs } from "../api/export-api";
import { useExportRuntimeStore } from "../stores/export-runtime-store";

const STATUS_POLL_INTERVAL_MS = 5000;
const STREAM_RECONNECT_DELAY_MS = 600;

/**
 * Global export runtime driver, mounted once at the app root so it runs on every
 * route (the inline export panel lives on /export, but the run must stay live and
 * visible from any tab). It:
 *   - hydrates `is_export_active` from the backend on mount / reload,
 *   - streams the worker log SSE into the store while a run is active, and
 *   - keeps that stream alive across the load -> export phase boundary so a
 *     transient `complete` between per-phase POSTs does not strand the panel on
 *     "Waiting for worker output..." (the premature-complete bug).
 *
 * The actual export sequence is driven by the store's `runExport` action (a
 * detached promise), so it survives navigation independently of this hook.
 */
export function useExportRuntimeLifecycle(): void {
  useEffect(() => {
    let disposed = false;
    let openingStream = false;
    let streamController: AbortController | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const store = useExportRuntimeStore;

    const clearReconnect = () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const stopStream = () => {
      clearReconnect();
      if (streamController) {
        streamController.abort();
        streamController = null;
      }
      store.getState().setConnected(false);
    };

    const ensureStream = async () => {
      if (
        disposed ||
        openingStream ||
        streamController ||
        !store.getState().isExporting
      ) {
        return;
      }

      clearReconnect();
      openingStream = true;
      const controller = new AbortController();
      streamController = controller;

      try {
        await streamExportLogs({
          signal: controller.signal,
          since: store.getState().lastSeq,
          onOpen: () => store.getState().setConnected(true),
          onEvent: (event) => {
            if (event.event === "log" && event.entry) {
              store.getState().appendLog(event.entry, event.id ?? undefined);
            }
            // `complete` / `error` end this connection (streamExportLogs returns);
            // the finally block reconnects while the run is still in flight.
          },
        });
      } catch {
        // fetch-level failure: fall through to the reconnect below.
      } finally {
        openingStream = false;
        if (streamController === controller) {
          streamController = null;
        }
        store.getState().setConnected(false);

        if (
          !disposed &&
          !controller.signal.aborted &&
          store.getState().isExporting
        ) {
          reconnectTimer = setTimeout(() => {
            void ensureStream();
          }, STREAM_RECONNECT_DELAY_MS);
        }
      }
    };

    const pollStatus = async () => {
      if (!hasAuthToken()) return;
      try {
        const status = await getExportStatus();
        if (disposed) return;
        store.getState().applyBackendStatus(status);
        if (store.getState().isExporting) {
          void ensureStream();
        }
      } catch {
        // ignore transient status failures
      }
    };

    // React to isExporting flipping (run start / terminal) without needing the
    // subscribeWithSelector middleware: the base subscribe fires on every change.
    let prevExporting = store.getState().isExporting;
    const unsubscribe = store.subscribe((state) => {
      if (state.isExporting === prevExporting) return;
      prevExporting = state.isExporting;
      if (state.isExporting) {
        void ensureStream();
      } else {
        stopStream();
      }
    });

    void pollStatus();
    if (store.getState().isExporting) {
      void ensureStream();
    }

    const statusTimer = setInterval(() => {
      void pollStatus();
    }, STATUS_POLL_INTERVAL_MS);

    return () => {
      disposed = true;
      clearInterval(statusTimer);
      unsubscribe();
      stopStream();
    };
  }, []);
}
