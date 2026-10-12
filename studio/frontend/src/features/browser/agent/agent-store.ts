// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { Question } from "./gate";

/** A question the agent is waiting on, shown in the panel's agent bar. */
export type AgentPrompt = Question & { id: number };

type BrowserAgentState = {
  /** The composer's Browser tool: offer the model the browser tools. */
  enabled: boolean;
  setEnabled: (enabled: boolean) => void;
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
      tabId: null,
      working: false,
      question: null,
      runKey: null,
      stoppedRun: null,
    }),
    {
      name: "unsloth_browser_agent",
      version: 1,
      partialize: (state) => ({ enabled: state.enabled }),
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

/** The agent bar's Stop: the running chat's actions end, and its question is declined. */
export function stopBrowserAgent(): void {
  const { runKey, question } = useBrowserAgentStore.getState();
  if (question) answers.get(question.id)?.(false);
  useBrowserAgentStore.setState({ stoppedRun: runKey, working: false });
}
