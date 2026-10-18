// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The three small voice-mode models served by the backend: is the turn over, what to
// say while thinking, what to do with speech that lands mid-think. Every call resolves
// null when the model is not served or does not answer in time, so the caller keeps
// its word-rule fallback. With the decision log on, each answer carries the log row's
// id and the loop reports what happened next through reportLayaOutcome.

import { authFetch } from "@/features/auth";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import type { FillerDecision } from "./thinking-filler";
import {
  publishFillerDecision,
  publishInterjectDecision,
  publishTurnDecision,
} from "./voice-decisions";
import { voiceTrace } from "./voice-trace";

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

/** The log fields for this call: on from the settings toggle, off for a chat that opted out. */
function logFields(): { log: boolean; thread_id: string | null } {
  const threadId = useChatRuntimeStore.getState().activeThreadId ?? null;
  const voice = useVoiceSettingsStore.getState();
  const log = voice.voiceLogEnabled && !(threadId && voice.voiceLogOptOutThreadIds.includes(threadId));
  return { log, thread_id: threadId };
}

/** Report what happened after a logged verdict. A null id (log off, model unanswered) is a no-op. */
export function reportLayaOutcome(
  id: string | null | undefined,
  outcome: string,
  detail?: Record<string, unknown>,
): void {
  if (!id) return;
  voiceTrace("laya.outcome", { id, outcome, ...detail });
  void authFetch("/api/inference/voice/outcome", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, outcome, detail: detail ?? null }),
  }).catch(() => {});
}

export interface JudgeAnswer {
  p: number;
  /** The log row, when this call was logged. */
  id: string | null;
}

/** P(the user has finished the turn), or null without the judge. */
export async function judgeTurnComplete(text: string): Promise<JudgeAnswer | null> {
  const fields = logFields();
  const r = await post<{ p_complete: number; id?: string | null }>("judge", { text, ...fields });
  if (typeof r?.p_complete !== "number") return null;
  publishTurnDecision(r.p_complete, r.p_complete >= JUDGE_SEND_THRESHOLD);
  if (fields.log) voiceTrace("laya.judge", { id: r.id ?? null, p: r.p_complete });
  return { p: r.p_complete, id: r.id ?? null };
}

export async function fillerFromModel(
  text: string,
  prevAssistantAskedQuestion: boolean,
): Promise<FillerDecision | null> {
  const fields = logFields();
  const r = await post<{ say: boolean; class: string; phrase: string; id?: string | null }>(
    "filler",
    { text, prev_q: prevAssistantAskedQuestion, ...fields },
  );
  if (!r) return null;
  publishFillerDecision(r.say ? r.phrase : "");
  if (fields.log) voiceTrace("laya.filler", { id: r.id ?? null, say: r.say, cls: r.class });
  const logId = r.id ?? null;
  return r.say
    ? { say: true, cls: r.class as FillerDecision["cls"], phrase: r.phrase, logId }
    : { say: false, cls: "none", phrase: "", logId };
}

export type InterjectAction = "restart" | "ignore" | "queue";

export interface InterjectAnswer {
  action: InterjectAction;
  id: string | null;
}

export async function interjectAction(
  prompt: string,
  utterance: string,
  elapsedS: number,
): Promise<InterjectAnswer | null> {
  const fields = logFields();
  const r = await post<{ action: InterjectAction; confidence: number; id?: string | null }>(
    "interject",
    { prompt, utterance, elapsed_s: elapsedS, ...fields },
  );
  if (!r?.action) return null;
  publishInterjectDecision(r.action, r.confidence ?? 0);
  if (fields.log) voiceTrace("laya.interject", { id: r.id ?? null, action: r.action });
  return { action: r.action, id: r.id ?? null };
}
