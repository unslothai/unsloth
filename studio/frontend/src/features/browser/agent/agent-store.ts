// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Question } from "./gate";
import {
  type AgentPermissions,
  DEFAULT_PERMISSIONS,
  type DownloadsRule,
  type HistoryAccess,
  MAX_SITE_RULES,
  type ScriptAccess,
  type SiteRule,
  checkedPermissions,
  ruleFor,
} from "./permissions";

export type AgentPrompt = Question & { id: number };

/** One chat's use of the browser tools; chats run theirs side by side, each in its own tab. */
export type AgentChat = {
  /** The tab the agent drives in this chat; the model can't reach any other. */
  tabId: string | null;
  working: boolean;
  question: AgentPrompt | null;
  /** The chat run the agent is acting for, until it ends; Stop applies to it. */
  runKey: string | null;
  /** The chat run the user stopped from the agent bar; its actions answer "stopped". */
  stoppedRun: string | null;
  /** When its last run ended (ms), for downloads that land after it. */
  endedAt: number;
};

export const IDLE_AGENT: AgentChat = { tabId: null, working: false, question: null, runKey: null, stoppedRun: null, endedAt: 0 };

type BrowserAgentState = {
  /** The composer's Browser tool: offer the model the browser tools. */
  enabled: boolean;
  setEnabled: (enabled: boolean) => void;
  permissions: AgentPermissions;
  setDefaultRule: (patch: Partial<SiteRule>) => void;
  setSiteRule: (site: string, rule: SiteRule | null) => void;
  historyAccess: HistoryAccess;
  setHistoryAccess: (access: HistoryAccess) => void;
  scriptAccess: ScriptAccess;
  setScriptAccess: (access: ScriptAccess) => void;
  /** By chat key (browser store `chat`). */
  chats: Record<string, AgentChat>;
};

export const useBrowserAgentStore = create<BrowserAgentState>()(
  persist(
    (set) => ({
      enabled: false,
      setEnabled: (enabled) => set({ enabled }),
      permissions: DEFAULT_PERMISSIONS,
      setDefaultRule: (patch) =>
        set((state) => ({
          permissions: { ...state.permissions, defaults: { ...state.permissions.defaults, ...patch } },
        })),
      setSiteRule: (site, rule) =>
        set((state) => {
          const { [site]: _old, ...sites } = state.permissions.sites;
          if (rule && Object.keys(sites).length >= MAX_SITE_RULES) return state;
          return { permissions: { ...state.permissions, sites: rule ? { ...sites, [site]: rule } : sites } };
        }),
      historyAccess: "ask",
      setHistoryAccess: (historyAccess) => set({ historyAccess }),
      scriptAccess: "ask",
      setScriptAccess: (scriptAccess) => set({ scriptAccess }),
      chats: {},
    }),
    {
      name: "unsloth_browser_agent",
      version: 2,
      // Version 1 kept only `enabled`; merge fills the rest in.
      migrate: (persisted) => persisted as BrowserAgentState,
      partialize: (state) => ({
        enabled: state.enabled,
        permissions: state.permissions,
        historyAccess: state.historyAccess,
        scriptAccess: state.scriptAccess,
      }),
      // Checked on the way in: storage another build wrote must never read as "allow".
      merge: (persisted, current) => {
        const saved = (persisted ?? {}) as Record<string, unknown>;
        return {
          ...current,
          enabled: saved.enabled === true,
          permissions: checkedPermissions(saved.permissions),
          historyAccess: saved.historyAccess === "allow" || saved.historyAccess === "never" ? saved.historyAccess : "ask",
          scriptAccess: saved.scriptAccess === "never" ? "never" : "ask",
        };
      },
    },
  ),
);

// A new chat's key -> its saved one, so a run that captured the old key still reaches its chat.
const adopted = new Map<string, string>();

export function liveChat(chat: string): string {
  let key = chat;
  for (let hops = 0; hops < 8 && adopted.has(key); hops++) key = adopted.get(key) ?? key;
  return key;
}

export function agentChat(chat: string): AgentChat {
  return useBrowserAgentStore.getState().chats[liveChat(chat)] ?? IDLE_AGENT;
}

export function patchAgentChat(chat: string, patch: Partial<AgentChat>): void {
  const key = liveChat(chat);
  useBrowserAgentStore.setState((state) => ({
    chats: { ...state.chats, [key]: { ...(state.chats[key] ?? IDLE_AGENT), ...patch } },
  }));
}

/** A new chat saved under its id keeps its run, tab, pending question and the action in flight. */
export function adoptAgentChat(from: string, to: string): void {
  if (from === to) return;
  adopted.set(from, to);
  const action = activeActions.get(from);
  activeActions.delete(from);
  if (action && !activeActions.has(to)) activeActions.set(to, action);
  useBrowserAgentStore.setState((state) => {
    const moved = state.chats[from];
    if (!moved) return state;
    const { [from]: _moved, ...chats } = state.chats;
    return { chats: chats[to] ? chats : { ...chats, [to]: moved } };
  });
}

const answers = new Map<number, (allow: boolean) => void>();
let nextQuestion = 1;

/** Ask the user in `chat`'s agent bar; false when they decline or the run ends first. */
export function askUser(chat: string, question: Question, signal?: AbortSignal): Promise<boolean> {
  return new Promise((resolve) => {
    const id = nextQuestion++;
    const finish = (allow: boolean) => {
      answers.delete(id);
      signal?.removeEventListener("abort", onAbort);
      if (agentChat(chat).question?.id === id) patchAgentChat(chat, { question: null });
      resolve(allow);
    };
    const onAbort = () => finish(false);
    if (signal?.aborted) {
      resolve(false);
      return;
    }
    signal?.addEventListener("abort", onAbort, { once: true });
    // One at a time per chat: its actions run in order, so an older question can only be a leftover.
    const current = agentChat(chat).question;
    if (current) answers.get(current.id)?.(false);
    answers.set(id, finish);
    patchAgentChat(chat, { question: { ...question, id } });
  });
}

export function answerQuestion(id: number, allow: boolean): void {
  answers.get(id)?.(allow);
}

const DOWNLOAD_STRICTNESS: Record<DownloadsRule, number> = { allow: 0, ask: 1, block: 2 };

export function stricterDownloadRule(a: DownloadsRule | null, b: DownloadsRule | null): DownloadsRule | null {
  if (!a || !b) return a ?? b;
  return DOWNLOAD_STRICTNESS[a] >= DOWNLOAD_STRICTNESS[b] ? a : b;
}

/** A blob: URL has no host of its own; the site that made it is its origin. */
function creator(url: string): string {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "blob:" ? parsed.origin : url;
  } catch {
    return url;
  }
}

/** The Downloads permission for a download from `tabId`, or null when that isn't the agent's tab.
 *  The site that started it decides, so a blocked site can't hand its files to an allowed CDN; the
 *  file's own host only when it has a rule of its own, and then the stricter one wins. */
export function agentDownloadRule(tabId: string, site: string, fileUrl: string): DownloadsRule | null {
  const state = useBrowserAgentStore.getState();
  const agent = Object.values(state.chats).find((chat) => chat.tabId === tabId);
  // Between runs the tab is the user's again, under their own download settings.
  if (!agent || (!agent.runKey && Date.now() - agent.endedAt > DOWNLOAD_GRACE_MS)) return null;
  const rule = ruleFor(creator(site || fileUrl), state.permissions).rule.downloads;
  const file = ruleFor(creator(fileUrl), state.permissions);
  if (!file.site) return rule;
  return DOWNLOAD_STRICTNESS[file.rule.downloads] > DOWNLOAD_STRICTNESS[rule] ? file.rule.downloads : rule;
}

const activeActions = new Map<string, AbortController>();

export function setActiveAction(chat: string, controller: AbortController | null): void {
  if (controller) activeActions.set(liveChat(chat), controller);
  else activeActions.delete(liveChat(chat));
}

// A download the agent started can be reported after its run ends (a slow response, a native
// view's prompt), so the agent's rule outlasts the run by this much.
const DOWNLOAD_GRACE_MS = 120_000;

const runEndListeners = new Set<(chat: string) => void>();

/** `listener(chat)` when a chat's run with the browser tools ends. */
export function onBrowserAgentRunEnd(listener: (chat: string) => void): () => void {
  runEndListeners.add(listener);
  return () => runEndListeners.delete(listener);
}

export function endBrowserAgentRun(runKey: string): void {
  for (const [chat, agent] of Object.entries(useBrowserAgentStore.getState().chats)) {
    if (agent.runKey !== runKey) continue;
    patchAgentChat(chat, { runKey: null, endedAt: Date.now() });
    for (const listener of runEndListeners) listener(chat);
  }
}

/** The agent bar's Stop: the action running now ends too, not just the ones after it. */
export function stopBrowserAgent(chat: string): void {
  const { runKey, question } = agentChat(chat);
  patchAgentChat(chat, { stoppedRun: runKey, working: false });
  if (question) answers.get(question.id)?.(false);
  activeActions.get(liveChat(chat))?.abort();
}
