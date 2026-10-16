// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The three small voice-mode models served by the backend: is the turn over, what to
// say while thinking, what to do with speech that lands mid-think. Every call resolves
// null when the model is not served or does not answer in time, so the caller keeps
// its word-rule fallback.

import { authFetch } from "@/features/auth";
import type { FillerDecision } from "./thinking-filler";

const TIMEOUT_MS = 400;

async function post<T>(path: string, body: unknown): Promise<T | null> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
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
}

/** P(the user has finished the turn), or null without the judge. */
export async function judgeTurnComplete(text: string): Promise<number | null> {
  const r = await post<{ p_complete: number }>("judge", { text });
  return typeof r?.p_complete === "number" ? r.p_complete : null;
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
  const r = await post<{ action: InterjectAction }>("interject", {
    prompt,
    utterance,
    elapsed_s: elapsedS,
  });
  return r?.action ?? null;
}
