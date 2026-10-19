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
    ["Checking something.", 3],
    ["One moment.", 2],
  ],
  elapsed: [
    ["Still thinking.", 4],
    ["This is a larger request, one moment please.", 3],
    ["Give me a second.", 2],
  ],
};

// Every spoken line before the answer is a thinking sound, never a yes, no or "okay": the
// model decides the answer, so a filler that sounds like agreement can contradict it.

// The classifier chose silence but the reply is slow anyway: a short bridge so the
// pause does not read as a miss. No bare "Hmm.": OmniVoice voices it ~8 dB under speech,
// a murmur that reads as room noise.
const QUIET_FALLBACK: Weighted = [
  ["Let me see.", 3],
  ["One sec.", 3],
  ["Let me think.", 2],
];

// The answer is being written and its first sentence is still on the way. It may be a
// refusal, so nothing here promises it ("almost there", "here we go").
const WRITING: Weighted = [
  ["One moment.", 4],
  ["Just a second.", 2],
];

// A tool call is paused on the Allow/Deny card in the chat: without a line the loop just
// goes quiet and looks stuck.
const APPROVAL: Weighted = [
  ["I need your permission to run that. Tap Allow in the chat.", 3],
  ["Can you allow that in the chat? I'll wait.", 2],
];

export function pickApproval(rand: () => number = Math.random): string {
  return pickWeighted(APPROVAL, rand);
}

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

// Nothing has been said aloud yet this turn: "still" would refer to nothing.
const FIRST_ELAPSED: Weighted = [
  ["Let me think.", 3],
  ["One moment.", 2],
];

export function pickStage2(
  toolKinds: readonly ToolKind[],
  rand: () => number = Math.random,
  first = false,
): string {
  const kind = toolKinds[toolKinds.length - 1];
  if (!kind && first) return pickWeighted(FIRST_ELAPSED, rand);
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
// "Still on it"-style asides wait this long after any aside: three in 25 s felt like nagging.
export const ASIDE_MIN_GAP_MS = 12000;

export function fillerSystemNote(phrase: string): string {
  return (
    `Voice mode: while you were thinking, the assistant already said aloud: "${phrase}" ` +
    "Continue naturally from that, as the same speaker. Do not repeat it. " +
    "If you cannot do what was asked, say so plainly after it."
  );
}

/** The user cut in to change direction: the earlier bubbles of this ask are withdrawn, not merged. */
export function replaceSystemNote(): string {
  return (
    "Voice mode: the user interrupted to cancel what they had asked and change direction. " +
    "Answer only their latest message; do not carry on with or refer back to the withdrawn ask."
  );
}

// Set by the voice loop at submit time, read once by the chat adapter when the
// run builds its request. The adapter does not know about voice otherwise.
let pendingFiller: FillerDecision | null = null;
let pendingReplace = false;

export function setPendingReplace(replace: boolean): void {
  pendingReplace = replace;
}

export function consumePendingReplace(): boolean {
  const r = pendingReplace;
  pendingReplace = false;
  return r;
}

export function setPendingFiller(decision: FillerDecision | null): void {
  pendingFiller = decision?.say ? decision : null;
}

export function consumePendingFiller(): FillerDecision | null {
  const d = pendingFiller;
  pendingFiller = null;
  return d;
}

/** Every fixed aside the loop can say, for warming the voice's cache while idle. */
/** Short, always-cached lines that can stand in for a filler not synthesized yet. */
export function fillerStandIns(): string[] {
  return [...FIRST_ELAPSED, ...QUIET_FALLBACK].map(([phrase]) => phrase);
}

export function asidePhrases(): string[] {
  const out: string[] = [];
  for (const bank of [QUIET_FALLBACK, WRITING, FIRST_ELAPSED, APPROVAL, ...Object.values(STAGE2)]) {
    for (const [phrase] of bank) if (!out.includes(phrase)) out.push(phrase);
  }
  return out;
}
