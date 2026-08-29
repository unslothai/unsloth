// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getApiMonitorEntry } from "../api/chat-api";
import {
  type LiveChatTpsEntry,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";
import {
  readLiveChatTpsSample,
  visibleLiveChatTps,
} from "../lib/live-chat-tps";
import { useEffect } from "react";

const POLL_INTERVAL_MS = 1500;

function latestEntry(entries: LiveChatTpsEntry[] | undefined) {
  return entries?.[entries.length - 1];
}

export function useLiveChatTps(): number | null {
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const threadKey = activeThreadId || "__default";
  const owner = useChatRuntimeStore(
    (state) => latestEntry(state.liveTpsByThreadId[threadKey])?.owner,
  );
  const monitorId = useChatRuntimeStore(
    (state) => latestEntry(state.liveTpsByThreadId[threadKey])?.monitorId,
  );
  const phase = useChatRuntimeStore(
    (state) => latestEntry(state.liveTpsByThreadId[threadKey])?.phase,
  );
  const lastRunningTps = useChatRuntimeStore(
    (state) =>
      latestEntry(state.liveTpsByThreadId[threadKey])?.lastRunningTps ?? null,
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
      } catch {
        if (cancelled || controller.signal.aborted) return;
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
