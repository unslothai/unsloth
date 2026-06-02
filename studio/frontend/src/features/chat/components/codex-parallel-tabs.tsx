// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Tabbed render for Codex parallel-calls fan-out.
 *
 * The backend emits four ``_toolEvent`` shapes for ``parallel_calls > 1``:
 *
 *   - ``codex_tab_open  {tab_id, query, total_tabs}``
 *   - ``codex_tab_chunk {tab_id, text}``
 *   - ``codex_tab_close {tab_id}``
 *   - ``codex_gather    {summary, tab_count}``
 *
 * The chat-adapter passes these events into ``useCodexParallelTabs``
 * via the shared tool-event channel. The hook collapses them into a
 * tab list (one entry per ``tab_id``) plus a synthesis row, and the
 * component below renders a horizontal tab strip with the active
 * tab's text in a scrollable panel below. The Synthesis tab is
 * highlighted because it's the unified answer the user usually wants
 * to read.
 */

import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";

export interface CodexTabState {
  /** 1-based tab index from the backend. */
  tabId: number;
  /** Accumulated text from ``codex_tab_chunk`` events for this tab. */
  text: string;
  /** True once the matching ``codex_tab_close`` event has arrived. */
  closed: boolean;
  /** Set when a ``codex_tab_error`` event was emitted for this tab. */
  error?: string;
}

export interface CodexParallelState {
  /** Per-tab streamed text, keyed by tabId, sorted ascending. */
  tabs: CodexTabState[];
  /** The original user query echoed on each tab_open event. */
  query: string | null;
  /** Final synthesis text from the ``codex_gather`` event. */
  synthesis: string | null;
  /** Total tabs reported on the first ``codex_tab_open`` event. */
  totalTabs: number;
}

export type CodexParallelEvent =
  | { type: "codex_tab_open"; tab_id: number; query?: string; total_tabs?: number }
  | { type: "codex_tab_chunk"; tab_id: number; text: string }
  | { type: "codex_tab_close"; tab_id: number }
  | { type: "codex_tab_error"; tab_id: number; error?: string }
  | { type: "codex_gather"; summary?: string; tab_count?: number };

/**
 * Pure reducer: given the prior parallel state and a single event,
 * return the new state. Kept as a standalone function so the chat-
 * adapter can drive it without re-rendering, and so it's trivially
 * unit-testable.
 */
export function reduceCodexParallelState(
  prev: CodexParallelState,
  event: CodexParallelEvent,
): CodexParallelState {
  switch (event.type) {
    case "codex_tab_open": {
      // Idempotent: re-opening an existing tab leaves it intact.
      const exists = prev.tabs.some((t) => t.tabId === event.tab_id);
      const tabs = exists
        ? prev.tabs
        : [
            ...prev.tabs,
            { tabId: event.tab_id, text: "", closed: false },
          ].sort((a, b) => a.tabId - b.tabId);
      return {
        ...prev,
        tabs,
        query: prev.query ?? event.query ?? null,
        totalTabs: event.total_tabs ?? Math.max(prev.totalTabs, event.tab_id),
      };
    }
    case "codex_tab_chunk": {
      const tabs = prev.tabs.map((t) =>
        t.tabId === event.tab_id ? { ...t, text: t.text + event.text } : t,
      );
      // Auto-create the slot if a chunk arrived before its open event
      // (shouldn't happen with the current backend ordering, but
      // defending against that race keeps the UI stable).
      if (!tabs.some((t) => t.tabId === event.tab_id)) {
        tabs.push({ tabId: event.tab_id, text: event.text, closed: false });
        tabs.sort((a, b) => a.tabId - b.tabId);
      }
      return { ...prev, tabs };
    }
    case "codex_tab_close": {
      const tabs = prev.tabs.map((t) =>
        t.tabId === event.tab_id ? { ...t, closed: true } : t,
      );
      return { ...prev, tabs };
    }
    case "codex_tab_error": {
      const tabs = prev.tabs.map((t) =>
        t.tabId === event.tab_id
          ? { ...t, closed: true, error: event.error }
          : t,
      );
      return { ...prev, tabs };
    }
    case "codex_gather": {
      return { ...prev, synthesis: event.summary ?? "" };
    }
    default: {
      return prev;
    }
  }
}

export const EMPTY_CODEX_PARALLEL_STATE: CodexParallelState = {
  tabs: [],
  query: null,
  synthesis: null,
  totalTabs: 0,
};

/** True when the state carries at least one observed event. */
export function hasCodexParallelContent(state: CodexParallelState): boolean {
  return state.tabs.length > 0 || state.synthesis !== null;
}

interface Props {
  state: CodexParallelState;
  /** Collapsed by default per spec; user clicks to expand. */
  defaultCollapsed?: boolean;
}

export function CodexParallelTabs({ state, defaultCollapsed = true }: Props) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const [activeTab, setActiveTab] = useState<number | "synthesis">("synthesis");

  // Whenever the synthesis arrives, switch to it automatically -- it's
  // the answer the user usually reads. Use a useMemo + effect-like
  // pattern via render-time check so we don't depend on extra hooks.
  // (A useEffect would also work; this stays lighter.)
  const effectiveActive = useMemo<number | "synthesis">(() => {
    if (state.synthesis && activeTab !== "synthesis") {
      return activeTab;
    }
    if (state.synthesis) {
      return "synthesis";
    }
    if (state.tabs.length > 0 && activeTab === "synthesis") {
      return state.tabs[0].tabId;
    }
    return activeTab;
  }, [state.synthesis, state.tabs, activeTab]);

  if (!hasCodexParallelContent(state)) {
    return null;
  }

  const totalSlots = state.totalTabs || state.tabs.length;

  return (
    <div
      className={cn(
        "codex-parallel-card my-2 rounded-md border bg-muted/30 p-2 text-sm",
      )}
    >
      <button
        type="button"
        className="flex w-full items-center justify-between gap-2 rounded px-1 py-1 text-left text-xs font-medium text-muted-foreground hover:bg-accent/50"
        onClick={() => setCollapsed((v) => !v)}
        aria-expanded={!collapsed}
      >
        <span>
          Codex parallel calls
          {totalSlots > 0 ? ` (${state.tabs.length}/${totalSlots})` : null}
          {state.synthesis ? " — synthesis ready" : ""}
        </span>
        <span aria-hidden>{collapsed ? "+" : "−"}</span>
      </button>
      {!collapsed && (
        <>
          <div className="mt-2 flex flex-wrap gap-1 border-b pb-2">
            {state.tabs.map((tab) => (
              <button
                key={tab.tabId}
                type="button"
                className={cn(
                  "rounded-t px-2 py-1 text-xs font-medium",
                  effectiveActive === tab.tabId
                    ? "bg-background text-foreground"
                    : "text-muted-foreground hover:bg-accent/50",
                  tab.error && "text-destructive",
                )}
                onClick={() => setActiveTab(tab.tabId)}
              >
                Tab {tab.tabId}
                {tab.error ? " (error)" : tab.closed ? "" : " …"}
              </button>
            ))}
            {state.synthesis !== null && (
              <button
                type="button"
                className={cn(
                  "rounded-t px-2 py-1 text-xs font-semibold",
                  effectiveActive === "synthesis"
                    ? "bg-primary/15 text-primary"
                    : "text-primary/70 hover:bg-primary/10",
                )}
                onClick={() => setActiveTab("synthesis")}
              >
                Synthesis
              </button>
            )}
          </div>
          <div className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap rounded bg-background/50 p-2 text-xs">
            {effectiveActive === "synthesis"
              ? state.synthesis || "(waiting for synthesis…)"
              : (state.tabs.find((t) => t.tabId === effectiveActive)?.text ||
                  "(waiting…)")}
          </div>
        </>
      )}
    </div>
  );
}
