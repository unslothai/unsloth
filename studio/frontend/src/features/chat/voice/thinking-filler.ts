// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the voice loop says while the model is still thinking. Stage 1 comes from
// the filler model before the run starts, so the model can be told it already
// said it; stage 2 is picked from what the run is doing when the wait drags on.

export type FillerClass = "minimal" | "task" | "question" | "creative" | "fix";

export interface FillerDecision {
  say: boolean;
  cls: FillerClass | "none";
  phrase: string;
  /** The voice decision log row this came from, when logging was on. */
  logId?: string | null;
}

type Weighted = readonly (readonly [string, number])[];

export type ToolKind = "search" | "code" | "file" | "tool";

const STAGE2: Record<ToolKind | "elapsed", Weighted> = {
  search: [
    ["Looking that up.", 4],
    ["Checking on that.", 2],
  ],
  code: [
    ["Running it real quick.", 4],
    ["Trying it out.", 2],
  ],
  file: [
    ["Checking the file.", 4],
    ["Reading through it.", 2],
  ],
  tool: [
    ["Working on it.", 3],
    ["One moment.", 2],
  ],
  elapsed: [
    ["Still on it.", 4],
    ["This is a larger request, one moment please.", 3],
    ["Give me a second.", 2],
  ],
};

// The classifier chose silence but the reply is slow anyway: a short bridge so the
// pause does not read as a miss.
const QUIET_FALLBACK: Weighted = [
  ["Hmm, okay.", 3],
  ["One sec.", 3],
  ["Let me think.", 2],
];

// The answer is being written and its first sentence is still on the way: said in
// that gap, which is when "almost there" is true.
const WRITING: Weighted = [
  ["Almost there.", 4],
  ["Here we go.", 2],
];

export function pickQuietFallback(rand: () => number = Math.random): string {
  return pickWeighted(QUIET_FALLBACK, rand);
}

export function pickWriting(rand: () => number = Math.random): string {
  return pickWeighted(WRITING, rand);
}

// Nothing has played this long after the run started: say something.
export const QUIET_FALLBACK_AFTER_MS = 2500;
// Text is arriving but the first sentence has not played this long after the last aside.
export const WRITING_AFTER_MS = 2500;

export function pickWeighted(
  options: Weighted,
  rand: () => number = Math.random,
): string {
  let total = 0;
  for (const [, w] of options) total += w;
  let r = rand() * total;
  for (const [text, w] of options) {
    r -= w;
    if (r < 0) return text;
  }
  return options[options.length - 1]?.[0] ?? "";
}

export function pickStage2(
  toolKinds: readonly ToolKind[],
  rand: () => number = Math.random,
): string {
  const kind = toolKinds[toolKinds.length - 1];
  return pickWeighted(STAGE2[kind ?? "elapsed"], rand);
}

export function toolKindOf(toolName: string): ToolKind {
  const n = toolName.toLowerCase();
  if (/search|web|fetch|browse|url/.test(n)) return "search";
  if (/code|python|exec|run|shell|bash|interpreter/.test(n)) return "code";
  if (/read|file|skill|knowledge|document|grep|glob/.test(n)) return "file";
  return "tool";
}

/** 300–700 ms: an instant filler sounds canned. */
export function fillerDelayMs(rand: () => number = Math.random): number {
  return 300 + Math.floor(rand() * 400);
}

export const STAGE2_AFTER_MS = 3500;

export function fillerSystemNote(phrase: string): string {
  return (
    `Voice mode: while you were thinking, the assistant already said aloud: "${phrase}" ` +
    "Continue naturally from that, as the same speaker. Do not repeat it. " +
    "If you cannot do what was asked, say so plainly after it."
  );
}

// Set by the voice loop at submit time, read once by the chat adapter when the
// run builds its request. The adapter does not know about voice otherwise.
let pendingFiller: FillerDecision | null = null;

export function setPendingFiller(decision: FillerDecision | null): void {
  pendingFiller = decision?.say ? decision : null;
}

export function consumePendingFiller(): FillerDecision | null {
  const d = pendingFiller;
  pendingFiller = null;
  return d;
}

/** Every fixed aside the loop can say, for warming the voice's cache while idle. */
export function asidePhrases(): string[] {
  const out: string[] = [];
  for (const bank of [QUIET_FALLBACK, WRITING, ...Object.values(STAGE2)]) {
    for (const [phrase] of bank) if (!out.includes(phrase)) out.push(phrase);
  }
  return out;
}
