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
/** `always`: allowed, and not asked again for this kind of action on this site in this chat. */
export type Answer = false | "once" | "always";

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
  /** The tab the agent drives; the model can't reach any other. */
  tabId: string | null;
  working: boolean;
  question: AgentPrompt | null;
  /** The chat run the agent is acting for, until it ends; Stop applies to it. */
  runKey: string | null;
  /** The chat run the user stopped from the agent bar; its actions answer "stopped". */
  stoppedRun: string | null;
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
      tabId: null,
      working: false,
      question: null,
      runKey: null,
      stoppedRun: null,
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

const answers = new Map<number, (answer: Answer) => void>();
let nextQuestion = 1;

/** Ask the user in the agent bar; false when they decline or the run ends first. */
export function askUser(question: Question, signal?: AbortSignal): Promise<Answer> {
  return new Promise((resolve) => {
    const id = nextQuestion++;
    const finish = (answer: Answer) => {
      answers.delete(id);
      signal?.removeEventListener("abort", onAbort);
      if (useBrowserAgentStore.getState().question?.id === id) useBrowserAgentStore.setState({ question: null });
      resolve(answer);
    };
    const onAbort = () => finish(false);
    if (signal?.aborted) {
      resolve(false);
      return;
    }
    signal?.addEventListener("abort", onAbort, { once: true });
    // One at a time: actions run in order, so an older question can only be a leftover.
    const current = useBrowserAgentStore.getState().question;
    if (current) answers.get(current.id)?.(false);
    answers.set(id, finish);
    useBrowserAgentStore.setState({ question: { ...question, id } });
  });
}

export function answerQuestion(id: number, allow: boolean, always = false): void {
  answers.get(id)?.(allow && (always ? "always" : "once"));
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
  // Between runs the tab is the user's again, under their own download settings.
  if (state.tabId !== tabId || (!state.runKey && Date.now() - runEndedAt > DOWNLOAD_GRACE_MS)) return null;
  const rule = ruleFor(creator(site || fileUrl), state.permissions).rule.downloads;
  const file = ruleFor(creator(fileUrl), state.permissions);
  if (!file.site) return rule;
  return DOWNLOAD_STRICTNESS[file.rule.downloads] > DOWNLOAD_STRICTNESS[rule] ? file.rule.downloads : rule;
}

let activeAction: AbortController | null = null;

export function setActiveAction(controller: AbortController | null): void {
  activeAction = controller;
}

// A download the agent started can be reported after its run ends (a slow response, a native
// view's prompt), so the agent's rule outlasts the run by this much.
const DOWNLOAD_GRACE_MS = 120_000;
let runEndedAt = 0;

export function endBrowserAgentRun(runKey: string): void {
  if (useBrowserAgentStore.getState().runKey !== runKey) return;
  runEndedAt = Date.now();
  useBrowserAgentStore.setState({ runKey: null });
}

/** The agent bar's Stop: the action running now ends too, not just the ones after it. */
export function stopBrowserAgent(): void {
  const { runKey, question } = useBrowserAgentStore.getState();
  useBrowserAgentStore.setState({ stoppedRun: runKey, working: false });
  if (question) answers.get(question.id)?.(false);
  activeAction?.abort();
}
