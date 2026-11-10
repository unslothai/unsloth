// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { hasAuthToken } from "@/features/auth";
import { useEffect } from "react";
import {
  getBenchmarkStatus,
  streamBenchmarkLogs,
} from "../api/benchmark-api";
import { useBenchmarkRuntimeStore } from "../stores/benchmark-runtime-store";

const STATUS_POLL_INTERVAL_MS = 5000;
const STREAM_RECONNECT_DELAY_MS = 600;

/**
 * Global benchmark runtime driver, mounted once at the app root. It:
 *   - hydrates `is_benchmark_active` from the backend on mount / reload,
 *   - opens a single long-lived SSE stream to `/api/benchmarks/evals/logs/stream`
 *     that replays the current run from the start and then tails live log
 *     lines, keeping the connection open until the panel closes, and
 *   - auto-reconnects the stream (resuming at the last received seq) if the
 *     connection drops while a run is in flight.
 *
 * The run POST itself blocks until the benchmark finishes, so this stream is
 * the only live log transport while a run is active; the `/status` poll below
 * drives phase transitions (running -> success / error / canceled).
 */
export function useBenchmarkRuntimeLifecycle(): void {
  useEffect(() => {
    let disposed = false;
    let openingStream = false;
    let streamController: AbortController | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const store = useBenchmarkRuntimeStore;

    const isPanelActive = () => {
      const state = store.getState();
      return state.isRunning || state.phase !== "idle";
    };

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
        !isPanelActive()
      ) {
        return;
      }

      clearReconnect();
      openingStream = true;
      const controller = new AbortController();
      streamController = controller;

      try {
        await streamBenchmarkLogs({
          signal: controller.signal,
          since: store.getState().lastSeq,
          onOpen: () => store.getState().setConnected(true),
          onEvent: (event) => {
            if (event.event === "log" && event.entry) {
              store.getState().appendLog(event.entry, event.id ?? undefined);
            }
          },
        });
      } catch {
        // fetch-level failure: fall through to the reconnect below.
      } finally {
        openingStream = false;
        if (streamController === controller) {
          streamController = null;
        }

        if (
          !disposed &&
          !controller.signal.aborted &&
          isPanelActive()
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
        const status = await getBenchmarkStatus();
        if (disposed) return;
        store.getState().applyBackendStatus(status);
        if (isPanelActive()) {
          void ensureStream();
        }
      } catch {
        // ignore transient status failures
      }
    };

    let prevActive = isPanelActive();
    const unsubscribe = store.subscribe((state) => {
      const active = state.isRunning || state.phase !== "idle";
      if (active === prevActive) return;
      prevActive = active;
      if (active) {
        void ensureStream();
      } else {
        stopStream();
      }
    });

    void pollStatus();
    if (isPanelActive()) {
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
