// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useBrowserStore } from "@/features/browser";
import { useEffect, useState } from "react";
import {
  SubagentsApiError,
  getSubagent,
  listSubagents,
} from "../api/subagents-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useResearchRunStore } from "../stores/research-run-store";
import { useSubagentsStore } from "../stores/subagents-store";
import type { SubagentRecord } from "../types/subagents";
import {
  SUBAGENT_DETAIL_POLL_MS,
  SUBAGENT_LIST_POLL_UNAVAILABLE_MS,
  anySubagentRunning,
  shouldAutoOpenSubagentPanel,
  subagentForDescription,
  subagentListPollInterval,
} from "../utils/subagents";

/**
 * Keeps the store's subagent list for `threadId` fresh: every 1.5 s while one runs or the thread
 * is generating, every 10 s otherwise. Stops on unmount or when the thread changes. A backend
 * without the route (404) or a dropped connection is not an error worth showing.
 */
export function useSubagentsPolling(
  threadId: string | null | undefined,
  enabled = true,
): void {
  const generating = useChatRuntimeStore((state) =>
    threadId ? Boolean(state.runningByThreadId[threadId]) : false,
  );
  const anyRunning = useSubagentsStore((state) =>
    threadId ? anySubagentRunning(state.byThread[threadId]) : false,
  );
  const intervalMs = subagentListPollInterval({
    anyRunning,
    threadGenerating: generating,
  });

  useEffect(() => {
    if (!threadId || !enabled) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();

    const tick = async () => {
      let delay = intervalMs;
      try {
        const records = await listSubagents(threadId, controller.signal);
        if (cancelled) return;
        const store = useSubagentsStore.getState();
        const prevCount = store.byThread[threadId]?.length;
        store.setList(threadId, records);
        if (
          shouldAutoOpenSubagentPanel({
            prevCount,
            next: records,
            alreadyAutoOpened: Boolean(store.autoOpenedThreadIds[threadId]),
          })
        ) {
          store.markAutoOpened(threadId);
          // Never take the side pane from a page, run settings or research run the user opened.
          if (
            !useBrowserStore.getState().open &&
            !useResearchRunStore.getState().openRunId &&
            !useChatRuntimeStore.getState().settingsPanelOpen
          ) {
            store.openPanel(threadId);
          }
        }
      } catch (error) {
        if (cancelled) return;
        if (error instanceof SubagentsApiError && error.status === 404) {
          delay = SUBAGENT_LIST_POLL_UNAVAILABLE_MS;
        }
      }
      if (!cancelled) timer = setTimeout(() => void tick(), delay);
    };
    void tick();

    return () => {
      cancelled = true;
      controller.abort();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [threadId, enabled, intervalMs]);

  // The panel belongs to one chat; leaving it must not leave another chat's panel open.
  useEffect(() => {
    const state = useSubagentsStore.getState();
    if (state.openThreadId !== null && state.openThreadId !== threadId) {
      state.closePanel();
    }
  }, [threadId]);
}

/**
 * The full record of one subagent: read once, then every second while it runs, and once more when
 * it stops so the transcript ends on its last item. Failures leave the last good copy in place.
 */
export function useSubagentDetailPolling(
  subagentId: string | null,
  running: boolean,
): SubagentRecord | undefined {
  const record = useSubagentsStore((state) =>
    subagentId ? state.detailById[subagentId] : undefined,
  );

  useEffect(() => {
    if (!subagentId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const controller = new AbortController();

    const tick = async () => {
      try {
        const next = await getSubagent(subagentId, controller.signal);
        if (cancelled) return;
        useSubagentsStore.getState().setDetail(next);
      } catch {
        // 404 or a dropped connection: keep what is on screen.
      }
      if (!cancelled && running) {
        timer = setTimeout(() => void tick(), SUBAGENT_DETAIL_POLL_MS);
      }
    };
    void tick();

    return () => {
      cancelled = true;
      controller.abort();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [subagentId, running]);

  return record;
}

/** Wall clock that ticks once a second while `active`, for elapsed times of running subagents. */
export function useSubagentClock(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const tick = () => setNow(Date.now());
    // Deferred, so a clock that was idle does not show its old reading for a whole second.
    const first = setTimeout(tick, 0);
    const timer = setInterval(tick, 1000);
    return () => {
      clearTimeout(first);
      clearInterval(timer);
    };
  }, [active]);
  return now;
}

/**
 * Shows the subagent panel for a thread, optionally on one subagent's detail view. The side pane
 * holds one thing at a time, so this also closes run settings and the research panel.
 */
export function openSubagentPanel(
  threadId: string,
  subagentId?: string | null,
): void {
  useChatRuntimeStore.getState().setSettingsPanelOpen(false);
  useResearchRunStore.getState().closePanel();
  useSubagentsStore.getState().openPanel(threadId, subagentId);
}

/**
 * Opens the panel from a spawn_subagent tool card. The card knows only the description, so an
 * ambiguous one opens the list rather than the wrong subagent.
 */
export function openSubagentPanelForDescription(
  threadId: string | null | undefined,
  description: string,
): void {
  const id = threadId ?? useChatRuntimeStore.getState().activeThreadId;
  if (!id) return;
  const subagent = subagentForDescription(
    useSubagentsStore.getState().byThread[id],
    description,
  );
  openSubagentPanel(id, subagent?.id ?? null);
}
