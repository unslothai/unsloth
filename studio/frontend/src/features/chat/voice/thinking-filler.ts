// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What the voice loop says while the model is still thinking. Stage 1 is picked
// from the utterance before the run starts, so the model can be told it already
// said it; stage 2 is picked from what the run is doing when the wait drags on.
// The picker here is a word-rule stand-in for the trained filler model.

export type FillerClass = "minimal" | "task" | "question" | "creative" | "fix";

export interface FillerDecision {
  say: boolean;
  cls: FillerClass | "none";
  phrase: string;
}

type Weighted = readonly (readonly [string, number])[];

// Hedged on purpose: the loop may not be able to do the thing, and the model
// continues from whatever was said here.
const PHRASES: Record<FillerClass, Weighted> = {
  minimal: [
    ["Hmm.", 4],
    ["Let's see.", 3],
    ["Okay, one sec.", 3],
    ["Thinking.", 2],
    ["Mm, hold on.", 2],
  ],
  task: [
    ["Let me check if I can do that.", 4],
    ["Let me see what I can do.", 4],
    ["Okay, give me a second on that one.", 2],
    ["Let me look into it.", 2],
  ],
  question: [
    ["Hmm, let me think about that.", 4],
    ["Good question, let me think.", 3],
    ["Let me think about that one.", 3],
  ],
  creative: [
    ["Okay, let me come up with something.", 4],
    ["Fun, let me think.", 3],
    ["Let me see what I can come up with.", 3],
  ],
  fix: [
    ["Ah, let me take a look.", 4],
    ["Okay, let me see what went wrong.", 3],
    ["Let me check that.", 3],
  ],
};

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
    ["Okay, this one's a bit bigger.", 3],
    ["Almost there.", 2],
  ],
};

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

const GREETING =
  /^(hi|hey|hello|yo|sup|good (morning|afternoon|evening)|thanks?|thank you|ok(ay)?|cool|nice|yes|yeah|yep|no|nope|bye|see ya)\b/i;
const CREATIVE =
  /\b(poem|story|song|joke|haiku|lyrics|name ideas?|slogan|rap|limerick|write (me )?(a|an|some) (poem|story|song|joke))\b/i;
const FIX =
  /\b(error|bug|broken|fail(ed|ing|s)?|crash(ed|es|ing)?|doesn'?t work|not working|wrong|fix|exception|traceback|won'?t)\b/i;
const TASK =
  /\b(write|build|make|create|generate|implement|code|script|program|plan|draft|set ?up|refactor|design|summari[sz]e|translate|convert|compare|outline)\b/i;
const QUESTION = /\b(why|how|what|which|explain|difference|should i|would|could|does|is it)\b/i;

export interface FillerContext {
  /** The assistant just asked something; a short answer needs no filler. */
  prevAssistantAskedQuestion?: boolean;
  rand?: () => number;
}

export function predictFiller(text: string, ctx: FillerContext = {}): FillerDecision {
  const t = text.trim();
  const words = t.split(/\s+/).filter(Boolean).length;
  const none: FillerDecision = { say: false, cls: "none", phrase: "" };
  if (words === 0) return none;
  if (ctx.prevAssistantAskedQuestion && words <= 8) return none;
  if (GREETING.test(t) && words <= 4) return none;
  if (words <= 2) return none;
  const rand = ctx.rand ?? Math.random;
  let cls: FillerClass;
  if (FIX.test(t)) cls = "fix";
  else if (CREATIVE.test(t)) cls = "creative";
  else if (TASK.test(t)) cls = "task";
  else if (QUESTION.test(t) && words >= 5) cls = "question";
  else if (words >= 5) cls = "minimal";
  else return none;
  // Medium asks get the short sound; the sentences are for clearly big ones.
  if (cls !== "minimal" && words < 7 && rand() < 0.5) cls = "minimal";
  return { say: true, cls, phrase: pickWeighted(PHRASES[cls], rand) };
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
