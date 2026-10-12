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
  type SiteRule,
  checkedPermissions,
  ruleFor,
} from "./permissions";

/** A question the agent is waiting on, shown in the panel's agent bar. */
export type AgentPrompt = Question & { id: number };

type BrowserAgentState = {
  /** The composer's Browser tool: offer the model the browser tools. */
  enabled: boolean;
  setEnabled: (enabled: boolean) => void;
  permissions: AgentPermissions;
  setDefaultRule: (patch: Partial<SiteRule>) => void;
  /** null removes the site's rule. */
  setSiteRule: (site: string, rule: SiteRule | null) => void;
  /** Whether the model may read the browser's history (the browser_history tool). */
  historyAccess: HistoryAccess;
  setHistoryAccess: (access: HistoryAccess) => void;
  /** The tab the agent drives; the model can't reach any other. */
  tabId: string | null;
  working: boolean;
  question: AgentPrompt | null;
  /** The chat run whose action is running, which Stop applies to. */
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
      }),
      // Checked on the way in: storage another build wrote must never read as "allow".
      merge: (persisted, current) => {
        const saved = (persisted ?? {}) as Record<string, unknown>;
        return {
          ...current,
          enabled: saved.enabled === true,
          permissions: checkedPermissions(saved.permissions),
          historyAccess: saved.historyAccess === "allow" || saved.historyAccess === "never" ? saved.historyAccess : "ask",
        };
      },
    },
  ),
);

const answers = new Map<number, (allow: boolean) => void>();
let nextQuestion = 1;

/** Ask the user in the agent bar; false when they decline or the run ends first. */
export function askUser(question: Question, signal?: AbortSignal): Promise<boolean> {
  return new Promise((resolve) => {
    const id = nextQuestion++;
    const finish = (allow: boolean) => {
      answers.delete(id);
      signal?.removeEventListener("abort", onAbort);
      if (useBrowserAgentStore.getState().question?.id === id) useBrowserAgentStore.setState({ question: null });
      resolve(allow);
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

export function answerQuestion(id: number, allow: boolean): void {
  answers.get(id)?.(allow);
}

const DOWNLOAD_STRICTNESS: Record<DownloadsRule, number> = { allow: 0, ask: 1, block: 2 };

/** The Downloads permission for a download from `tabId`, or null when that isn't the agent's tab.
 *  The site that started it decides, so a blocked site can't hand its files to an allowed CDN; the
 *  file's own host only when it has a rule of its own, and then the stricter one wins. */
export function agentDownloadRule(tabId: string, site: string, fileUrl: string): DownloadsRule | null {
  const state = useBrowserAgentStore.getState();
  if (state.tabId !== tabId) return null;
  const rule = ruleFor(site || fileUrl, state.permissions).rule.downloads;
  const file = ruleFor(fileUrl, state.permissions);
  if (!file.site) return rule;
  return DOWNLOAD_STRICTNESS[file.rule.downloads] > DOWNLOAD_STRICTNESS[rule] ? file.rule.downloads : rule;
}

let activeAction: AbortController | null = null;

export function setActiveAction(controller: AbortController | null): void {
  activeAction = controller;
}

/** The agent bar's Stop: the action running now ends too, not just the ones after it. */
export function stopBrowserAgent(): void {
  const { runKey, question } = useBrowserAgentStore.getState();
  useBrowserAgentStore.setState({ stoppedRun: runKey, working: false });
  if (question) answers.get(question.id)?.(false);
  activeAction?.abort();
}
