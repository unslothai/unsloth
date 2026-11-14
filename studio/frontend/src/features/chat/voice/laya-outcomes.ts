// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


export const QUICK_CONTINUE_MS = 1500;
/** An ignored or queued utterance said again inside this means the verdict was wrong. */
export const REPEAT_WINDOW_MS = 15000;

export type JudgeOutcome = "sent" | "continued" | "continued_quickly" | "flushed";
export type InterjectOutcome = "ok" | "repeated" | "reasked";
export type FillerOutcome = "spoken" | "echoed" | "too_late" | "cut";

const STOP = new Set(
  "a an the to of in on for and or but so um uh like you know i me my we our it its this that is are was were be do does did can could would should will please just really very with at by from as about what yeah yes no ok okay oh hmm well right sure".split(
    " ",
  ),
);

export function contentWords(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s']/gu, " ")
      .split(/\s+/)
      .filter((w) => w && !STOP.has(w)),
  );
}

export function sharesMostWords(earlier: string, later: string): boolean {
  const a = contentWords(earlier);
  if (a.size === 0) return false;
  const b = contentWords(later);
  let hit = 0;
  for (const w of a) if (b.has(w)) hit++;
  return hit / a.size >= 0.6;
}

const CORRECTION_CUE = /^\W*(?:no|nope|actually|instead|i mean|sorry|rather|wait)\b/i;

/**
 * The same request again: adds no word of its own ("five minutes" for "ten minutes"), keeps the
 * order of the words it shares ("beta into alpha" reverses "alpha into beta") and opens with no
 * correction cue ("No, ...", "Actually ...").
 */
const NEGATOR = /\b(?:no|not|never|none|nothing|without|\w+n['’]t)\b/gi;
const negations = (text: string) => (text.match(NEGATOR) ?? []).length;

export function isSameAsk(earlier: string, later: string): boolean {
  if (CORRECTION_CUE.test(later)) return false;
  // "with no balances" for "with balances": the stop list drops the one word that changed.
  if (negations(earlier) !== negations(later)) return false;
  const before = contentWords(earlier);
  const after = contentWords(later);
  if (!sharesMostWords(earlier, later) || ![...after].every((w) => before.has(w))) return false;
  const order = [...before].filter((w) => after.has(w));
  return order.join(" ") === [...after].join(" ");
}

const HOLD_ON =
  /^(?:(?:um|uh|oh|okay|ok|so|no|hey) )*(?:wait(?: wait)*|hold on|hang on|one sec(?:ond)?|one moment|just a sec(?:ond)?|give me a (?:sec(?:ond)?|minute|moment)|give me two minutes)$/;

export function isHoldOn(text: string): boolean {
  return HOLD_ON.test(normalizePhrase(text));
}

function normalizePhrase(text: string): string {
  return text.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, " ").replace(/\s+/g, " ").trim();
}

export function replyEchoesFiller(reply: string, phrase: string): boolean {
  const p = normalizePhrase(phrase);
  if (!p) return false;
  const head = normalizePhrase(reply.slice(0, Math.max(200, phrase.length * 4)));
  return head.includes(p);
}

export function interjectOutcomeFor(
  verdict: "ignore" | "queue" | "restart",
  earlier: string,
  later: string | null,
  gapMs: number,
): InterjectOutcome {
  if (verdict === "restart" || later === null || gapMs > REPEAT_WINDOW_MS) return "ok";
  if (!sharesMostWords(earlier, later)) return "ok";
  return verdict === "queue" ? "reasked" : "repeated";
}

export function interjectUnderApproval<A extends "ignore" | "queue" | "restart" | null>(
  action: A,
  awaitingApproval: boolean,
): A | "restart" {
  return action === "queue" && awaitingApproval ? "restart" : action;
}
