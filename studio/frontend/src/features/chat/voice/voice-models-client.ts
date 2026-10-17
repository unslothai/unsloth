// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The three small voice-mode models served by the backend: is the turn over, what to
// say while thinking, what to do with speech that lands mid-think. Every call resolves
// null when the model is not served or does not answer in time, so the caller keeps
// its word-rule fallback.

import { authFetch } from "@/features/auth";
import type { FillerDecision } from "./thinking-filler";
import {
  publishFillerDecision,
  publishInterjectDecision,
  publishTurnDecision,
} from "./voice-decisions";

const TIMEOUT_MS = 400;
/** 0.65 trades a few more waits for fewer cut-offs; the sweep is in the model card. */
export const JUDGE_SEND_THRESHOLD = 0.65;

async function post<T>(path: string, body: unknown): Promise<T | null> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
  const call = (async () => {
    try {
      const res = await authFetch(`/api/inference/voice/${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: ctl.signal,
      });
      if (!res.ok) return null;
      return (await res.json()) as T;
    } catch {
      return null;
    } finally {
      clearTimeout(timer);
    }
  })();
  // A fetch the abort never reaches would strand the turn behind it (the reply stays
  // held, nothing is sent), so the caller gets its answer on the clock either way.
  let settle: ReturnType<typeof setTimeout> | undefined;
  const late = new Promise<null>((r) => {
    settle = setTimeout(() => r(null), TIMEOUT_MS * 2);
  });
  return Promise.race([call, late]).finally(() => clearTimeout(settle));
}

/** P(the user has finished the turn), or null without the judge. */
export async function judgeTurnComplete(text: string): Promise<number | null> {
  const r = await post<{ p_complete: number }>("judge", { text });
  if (typeof r?.p_complete !== "number") return null;
  publishTurnDecision(r.p_complete, r.p_complete >= JUDGE_SEND_THRESHOLD);
  return r.p_complete;
}

export async function fillerFromModel(
  text: string,
  prevAssistantAskedQuestion: boolean,
): Promise<FillerDecision | null> {
  const r = await post<{ say: boolean; class: string; phrase: string }>("filler", {
    text,
    prev_q: prevAssistantAskedQuestion,
  });
  if (!r) return null;
  publishFillerDecision(r.say ? r.phrase : "");
  return r.say
    ? { say: true, cls: r.class as FillerDecision["cls"], phrase: r.phrase }
    : { say: false, cls: "none", phrase: "" };
}

export type InterjectAction = "restart" | "ignore" | "queue";

export async function interjectAction(
  prompt: string,
  utterance: string,
  elapsedS: number,
): Promise<InterjectAction | null> {
  const r = await post<{ action: InterjectAction; confidence: number }>("interject", {
    prompt,
    utterance,
    elapsed_s: elapsedS,
  });
  if (!r?.action) return null;
  publishInterjectDecision(r.action, r.confidence ?? 0);
  return r.action;
}
