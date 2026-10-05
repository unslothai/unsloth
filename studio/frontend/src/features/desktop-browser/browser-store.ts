// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { BrowserSession } from "./browser-session";

const AGENT_ENABLED_KEY = "unsloth_chat_browser_tools_enabled";

function readAgentEnabled(): boolean {
  try {
    return globalThis.localStorage?.getItem(AGENT_ENABLED_KEY) === "true";
  } catch {
    return false;
  }
}

function writeAgentEnabled(value: boolean): void {
  try {
    globalThis.localStorage?.setItem(AGENT_ENABLED_KEY, String(value));
  } catch {
    // keep the in-memory choice when storage is unavailable.
  }
}

export type BrowserApprovalChoice = "allow" | "allow-site" | "deny";

/** a question the agent is waiting on, shown on the tool card that asked it. */
export type BrowserApprovalRequest = {
  question: string;
  detail: string | null;
  /** the site an "allow on this site" answer covers, or null when the action is always asked. */
  site: string | null;
  resolve: (choice: BrowserApprovalChoice) => void;
};

export type BrowserActivity = {
  threadId: string | null;
  label: string;
  stop?: () => void;
  /** set while the agent waits for the user to finish something in the page. */
  done?: () => void;
};

export type BrowserHandoff = {
  reason: string;
  done: () => void;
};

type BrowserStoreState = {
  /** the pane is wanted beside the chat; the chat page owns whether it can show. */
  open: boolean;
  /** the chat page can show the pane right now (desktop, single chat, no canvas or research). */
  available: boolean;
  /** the chat the pane sits beside, when it has an id yet. */
  viewThreadId: string | null;
  /** the chat on screen: its id, or `new:<nonce>` for a new chat that has none yet. */
  viewKey: string | null;
  // the key this view started as, kept while a new chat adopts its id in place
  viewOrigin: string | null;
  /** the composer's Browser pill: the model is offered the browser tools. */
  agentEnabled: boolean;
  session: BrowserSession | null;
  activity: BrowserActivity | null;
  /** keyed by the tool card's part id. */
  approvals: Record<string, BrowserApprovalRequest>;
  /** keyed by the tool card's part id. */
  handoffs: Record<string, BrowserHandoff>;
  /** sites the user let the agent act on, per chat session. */
  allowedSites: Record<string, string[]>;
  setOpen: (open: boolean) => void;
  setAvailable: (
    available: boolean,
    viewThreadId: string | null,
    viewKey: string | null,
    viewOrigin: string | null,
  ) => void;
  setAgentEnabled: (enabled: boolean) => void;
  attachSession: (session: BrowserSession | null) => void;
  setActivity: (activity: BrowserActivity | null) => void;
  setApproval: (partId: string, request: BrowserApprovalRequest | null) => void;
  setHandoff: (partId: string, handoff: BrowserHandoff | null) => void;
  allowSite: (sessionKey: string, site: string) => void;
};

export const useDesktopBrowserStore = create<BrowserStoreState>()((set) => ({
  open: false,
  available: false,
  viewThreadId: null,
  viewKey: null,
  viewOrigin: null,
  agentEnabled: readAgentEnabled(),
  session: null,
  activity: null,
  approvals: {},
  handoffs: {},
  allowedSites: {},
  setOpen: (open) => set({ open }),
  setAvailable: (available, viewThreadId, viewKey, viewOrigin) =>
    set({ available, viewThreadId, viewKey, viewOrigin }),
  setAgentEnabled: (agentEnabled) => {
    writeAgentEnabled(agentEnabled);
    set({ agentEnabled });
  },
  attachSession: (session) => set({ session }),
  setActivity: (activity) => set({ activity }),
  setApproval: (partId, request) =>
    set((state) => {
      const approvals = { ...state.approvals };
      if (request) approvals[partId] = request;
      else delete approvals[partId];
      return { approvals };
    }),
  setHandoff: (partId, handoff) =>
    set((state) => {
      const handoffs = { ...state.handoffs };
      if (handoff) handoffs[partId] = handoff;
      else delete handoffs[partId];
      return { handoffs };
    }),
  allowSite: (sessionKey, site) =>
    set((state) => {
      const sites = state.allowedSites[sessionKey] ?? [];
      if (sites.includes(site)) return {};
      return {
        allowedSites: { ...state.allowedSites, [sessionKey]: [...sites, site] },
      };
    }),
}));

/** whether a run sent from `viewKey` (and `threadId`, once it had one) still belongs to the chat on screen */
export function isRunsChatOnScreen(
  threadId: string | null,
  viewKey: string | null,
): boolean {
  const state = useDesktopBrowserStore.getState();
  if (threadId && state.viewThreadId) return state.viewThreadId === threadId;
  // a run without an id yet, or sent from a new chat, matches the view it was sent from
  return (
    viewKey !== null &&
    (viewKey === state.viewKey || viewKey === state.viewOrigin)
  );
}

/** a send offers the browser tools when the pill is on and the pane can show beside that chat; a chat without an id is the new one on screen. */
export function desktopBrowserToolsOn(threadId: string | null): boolean {
  const { agentEnabled, available, viewThreadId } =
    useDesktopBrowserStore.getState();
  if (!agentEnabled || !available) return false;
  return !threadId || !viewThreadId || threadId === viewThreadId;
}

/** the ready session, opening the pane first if it is closed; rejects when the pane cannot show. */
export function waitForBrowserSession(
  timeoutMs = 15_000,
  signal?: AbortSignal,
): Promise<BrowserSession> {
  const store = useDesktopBrowserStore;
  const ready = () => {
    const session = store.getState().session;
    return session?.isReady ? session : null;
  };
  const existing = ready();
  if (existing) return Promise.resolve(existing);
  if (!store.getState().available) {
    return Promise.reject(
      new Error(
        "The browser can only open beside a single chat in the desktop app",
      ),
    );
  }
  store.getState().setOpen(true);
  return new Promise((resolve, reject) => {
    let unsubscribeSession: (() => void) | null = null;
    const finish = (error: Error | null, session?: BrowserSession) => {
      window.clearTimeout(timer);
      unsubscribeStore();
      unsubscribeSession?.();
      signal?.removeEventListener("abort", onAbort);
      if (error) reject(error);
      else if (session) resolve(session);
    };
    const check = () => {
      const session = ready();
      if (session) finish(null, session);
    };
    const watch = (session: BrowserSession | null) => {
      unsubscribeSession?.();
      unsubscribeSession = session ? session.subscribe(check) : null;
      check();
    };
    const onAbort = () => finish(new Error("Stopped"));
    const timer = window.setTimeout(
      () => finish(new Error("The browser did not open in time")),
      timeoutMs,
    );
    const unsubscribeStore = store.subscribe((state, previous) => {
      if (state.session !== previous.session) watch(state.session);
      if (!state.available && previous.available) {
        finish(new Error("The browser was closed"));
      }
    });
    signal?.addEventListener("abort", onAbort, { once: true });
    watch(store.getState().session);
  });
}
