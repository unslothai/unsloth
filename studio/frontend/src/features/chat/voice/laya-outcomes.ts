// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The small facts the loop reports back to the voice decision log, kept free of
// imports so they can be tested on node. Which outcome contradicts which verdict
// is decided on the backend (voice_log.ERROR_KINDS); this side only observes.

/** A "finished" verdict followed by more speech inside this is a cut-off turn. */
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

/** True when most of the earlier utterance's content words come back in the later one. */
export function sharesMostWords(earlier: string, later: string): boolean {
  const a = contentWords(earlier);
  if (a.size === 0) return false;
  const b = contentWords(later);
  let hit = 0;
  for (const w of a) if (b.has(w)) hit++;
  return hit / a.size >= 0.6;
}

function normalizePhrase(text: string): string {
  return text.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, " ").replace(/\s+/g, " ").trim();
}

/** Whether the reply repeated the filler phrase it was told it had already said. */
export function replyEchoesFiller(reply: string, phrase: string): boolean {
  const p = normalizePhrase(phrase);
  if (!p) return false;
  const head = normalizePhrase(reply.slice(0, Math.max(200, phrase.length * 4)));
  return head.includes(p);
}

/** What the loop reports for an interrupt verdict once the next utterance (or the window's end) is known. */
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
