// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  getApiMonitorEntry,
  isPermanentApiMonitorEntryError,
} from "../api/chat-monitor";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import {
  liveChatTpsThreadKey,
  newestRunningLiveChatTpsEntry,
  readLiveChatTpsSample,
  visibleLiveChatTps,
} from "../lib/live-chat-tps";
import { useEffect } from "react";

const POLL_INTERVAL_MS = 1500;

export function useLiveChatTps(routedThreadId?: string): number | null {
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const threadKey = liveChatTpsThreadKey(routedThreadId, activeThreadId);
  const owner = useChatRuntimeStore(
    (state) =>
      newestRunningLiveChatTpsEntry(state.liveTpsByThreadId[threadKey])?.owner,
  );
  const monitorId = useChatRuntimeStore(
    (state) =>
      newestRunningLiveChatTpsEntry(state.liveTpsByThreadId[threadKey])
        ?.monitorId,
  );
  const phase = useChatRuntimeStore(
    (state) =>
      newestRunningLiveChatTpsEntry(state.liveTpsByThreadId[threadKey])?.phase,
  );
  const lastRunningTps = useChatRuntimeStore(
    (state) =>
      newestRunningLiveChatTpsEntry(state.liveTpsByThreadId[threadKey])
        ?.lastRunningTps ?? null,
  );

  useEffect(() => {
    if (!owner || !monitorId || phase !== "running") return;

    const controller = new AbortController();
    let cancelled = false;
    let timer: number | undefined;

    const finish = () => {
      const store = useChatRuntimeStore.getState();
      store.finishThreadLiveTps(
        store.runKeyForOwner(threadKey, owner),
        owner,
      );
    };

    const poll = async (): Promise<void> => {
      try {
        const entry = await getApiMonitorEntry(monitorId, controller.signal);
        if (cancelled) return;
        const sample = readLiveChatTpsSample(entry, monitorId);
        if (!sample.running) {
          finish();
          return;
        }
        if (sample.tps !== null) {
          const store = useChatRuntimeStore.getState();
          store.setThreadLiveTps(
            store.runKeyForOwner(threadKey, owner),
            owner,
            monitorId,
            sample.tps,
          );
        }
      } catch (error) {
        if (cancelled || controller.signal.aborted) return;
        const store = useChatRuntimeStore.getState();
        store.clearThreadLiveTpsSample(
          store.runKeyForOwner(threadKey, owner),
          owner,
          monitorId,
        );
        if (isPermanentApiMonitorEntryError(error)) {
          finish();
          return;
        }
      }
      if (!cancelled) {
        timer = window.setTimeout(poll, POLL_INTERVAL_MS);
      }
    };

    void poll();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [monitorId, owner, phase, threadKey]);

  return visibleLiveChatTps(phase, lastRunningTps);
}
