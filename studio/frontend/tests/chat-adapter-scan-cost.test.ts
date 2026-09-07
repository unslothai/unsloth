// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { createSegmentedAssistantText } from "../src/features/chat/utils/incremental-assistant-content.ts";
import {
  createThinkTagTracker,
  hasUnclosedThinkTag,
  parseAssistantContent,
} from "../src/features/chat/utils/parse-assistant-content.ts";
import {
  createTrailingPlaceholderWatch,
  stripTrailingTemplatePlaceholder,
} from "../src/features/chat/utils/trailing-template-placeholder.ts";

/**
 * Complexity tests, not timing tests: they count the characters the scans look
 * at, so they say the same thing on an idle laptop and a loaded CI box. Every
 * string primitive the scans use is counted by what it can read:
 *
 *   String replace / lastIndexOf / indexOf   the receiver's length
 *   String slice                             the length it produces
 *   String startsWith                        the length it compares
 *   RegExp exec / test                       the length of the subject
 *
 * `test` is in the list because the bounded strip walks the trailing
 * whitespace run one character at a time, and the placeholder watch walks each
 * delta the same way; each of those reads shows up as a one-character subject,
 * so a walk that ran away would be counted, not hidden.
 */
type Counted = { chars: number };

function withScanAccounting<T>(body: () => T): { result: T; chars: number } {
  const counted: Counted = { chars: 0 };
  const realReplace = String.prototype.replace;
  const realLastIndexOf = String.prototype.lastIndexOf;
  const realIndexOf = String.prototype.indexOf;
  const realSlice = String.prototype.slice;
  const realStartsWith = String.prototype.startsWith;
  const realExec = RegExp.prototype.exec;
  const realTest = RegExp.prototype.test;

  type Unknown = (...args: unknown[]) => unknown;
  const countReceiver = (real: Unknown): Unknown =>
    function counting(this: string, ...args: unknown[]) {
      counted.chars += this.length;
      return real.apply(this, args);
    };

  String.prototype.replace = countReceiver(
    realReplace as unknown as Unknown,
  ) as unknown as typeof realReplace;
  String.prototype.lastIndexOf = countReceiver(
    realLastIndexOf as unknown as Unknown,
  ) as unknown as typeof realLastIndexOf;
  String.prototype.indexOf = countReceiver(
    realIndexOf as unknown as Unknown,
  ) as unknown as typeof realIndexOf;
  String.prototype.slice = function countingSlice(
    this: string,
    start?: number,
    end?: number,
  ): string {
    const out = realSlice.call(this, start, end);
    counted.chars += out.length;
    return out;
  };
  String.prototype.startsWith = function countingStartsWith(
    this: string,
    search: string,
    position?: number,
  ): boolean {
    counted.chars += String(search).length;
    return realStartsWith.call(this, search, position);
  } as unknown as typeof realStartsWith;
  RegExp.prototype.exec = function countingExec(
    this: RegExp,
    subject: string,
  ): RegExpExecArray | null {
    counted.chars += String(subject).length;
    return realExec.call(this, subject);
  };
  RegExp.prototype.test = function countingTest(
    this: RegExp,
    subject: string,
  ): boolean {
    counted.chars += String(subject).length;
    return realTest.call(this, subject);
  };

  try {
    const result = body();
    return { result, chars: counted.chars };
  } finally {
    String.prototype.replace = realReplace;
    String.prototype.lastIndexOf = realLastIndexOf;
    String.prototype.indexOf = realIndexOf;
    String.prototype.slice = realSlice;
    String.prototype.startsWith = realStartsWith;
    RegExp.prototype.exec = realExec;
    RegExp.prototype.test = realTest;
  }
}

// --------------------------------------------------------------- fixture ---

const WORDS = [
  "the", "model", "streams", "an", "answer", "one", "token", "at", "a", "time",
  "so", "the", "adapter", "sees", "the", "whole", "reply", "again", "on",
  "every", "arrival", "which", "is", "what", "makes", "the", "tail", "of", "a",
  "long", "reply", "slower", "than", "its", "head",
];

function prose(chars: number, seed: number): string {
  let out = "";
  let value = seed;
  while (out.length < chars) {
    value = (value * 1103515245 + 12345) & 0x7fffffff;
    out += WORDS[value % WORDS.length];
    out += (value & 31) === 0 ? ".\n\n" : " ";
  }
  return out.slice(0, chars);
}

function code(chars: number): string {
  let out = "```ts\n";
  let index = 0;
  while (out.length < chars) {
    out += `export function step${index}(input: Record<string, number>) {\n`;
    out += `  return { ...input, n: ${index} };\n}\n\n`;
    index += 1;
  }
  return `${out.slice(0, Math.max(7, chars - 4))}\n\`\`\`\n`;
}

/** A reply shaped like a reasoning model's: think block, prose, code. */
function buildReply(chars: number): string {
  const think = `<think>${prose(Math.round(chars * 0.25), 1)}</think>`;
  const body = prose(Math.round(chars * 0.5), 2);
  return `${think}\n\n${body}\n\n${code(Math.round(chars * 0.25))}`;
}

function arrivalsOf(text: string, size = 4): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) {
    out.push(text.slice(at, at + size));
  }
  return out;
}

// ------------------------------------------------------------- the scans ---

/** What the adapter ran before: the whole buffer, on every arrival. */
const UNBOUNDED = /\s*\$\{[^}]*\}\s*$/;

function runStrip(
  arrivals: string[],
  strip: (text: string) => string,
): number {
  let text = "";
  for (const chunk of arrivals) {
    text += chunk;
    text = strip(text);
  }
  return text.length;
}

function runThink(
  arrivals: string[],
  ask: (text: string) => boolean,
): number {
  let text = "";
  let unclosed = 0;
  for (const chunk of arrivals) {
    text += chunk;
    if (ask(text)) unclosed += 1;
  }
  return unclosed;
}

/**
 * Everything the adapter does per arrival, in the order it does it: extend the
 * reply, decide whether a trailing fragment needs stripping, ask whether the
 * text ends inside a think block, and rebuild the content parts.
 *
 * The point of this shape is that it is driven by the deltas, exactly as the
 * adapter is. A version of any of these that wanted the whole buffer would
 * show up as a cost that grows with the square of the reply.
 */
function runArrivalPath(arrivals: string[]): number {
  const thinkTags = createThinkTagTracker();
  const watch = createTrailingPlaceholderWatch();
  const segmented = createSegmentedAssistantText();
  let text = "";
  let sink = 0;
  for (const chunk of arrivals) {
    text += chunk;
    thinkTags.append(chunk);
    watch.append(chunk);
    segmented.appendText(chunk);
    if (watch.isCandidate()) {
      const stripped = stripTrailingTemplatePlaceholder(text);
      if (stripped.length !== text.length) {
        text = stripped;
        thinkTags.retract(text);
        watch.retract(text);
      }
    }
    if (thinkTags.endsInsideThink()) sink += 1;
    sink += segmented.runs(text, []).length;
  }
  return sink;
}

/** The same per arrival, the way it was done before: reparse the whole reply. */
function runArrivalPathRereading(arrivals: string[]): number {
  let text = "";
  let sink = 0;
  for (const chunk of arrivals) {
    text += chunk;
    text = stripTrailingTemplatePlaceholder(text);
    if (hasUnclosedThinkTag(text)) sink += 1;
    sink += parseAssistantContent(text).length;
  }
  return sink;
}

const SMALL = 16_000;
const LARGE = 32_000;

function stripCost(chars: number, strip: (text: string) => string): number {
  const arrivals = arrivalsOf(buildReply(chars));
  return withScanAccounting(() => runStrip(arrivals, strip)).chars;
}

function thinkCost(chars: number, ask: () => (text: string) => boolean): number {
  const arrivals = arrivalsOf(buildReply(chars));
  const asker = ask();
  return withScanAccounting(() => runThink(arrivals, asker)).chars;
}

// ------------------------------------------------------------------ tests ---

test("the trailing placeholder strip is linear in the reply length", () => {
  const small = stripCost(SMALL, stripTrailingTemplatePlaceholder);
  const large = stripCost(LARGE, stripTrailingTemplatePlaceholder);
  const growth = large / small;

  // Twice the reply, twice the arrivals, so a scan whose per-arrival cost is
  // independent of what came before doubles. Near 4 means the whole buffer is
  // being read again on every arrival.
  assert.equal(
    growth < 2.5,
    true,
    `strip cost grew ${growth.toFixed(2)}x for twice the reply (${small} -> ${large} chars scanned); that is not linear`,
  );

  // The defect it replaces, measured the same way, so the test states what it
  // guards against rather than only asserting a number.
  const unboundedSmall = stripCost(SMALL, (text) => text.replace(UNBOUNDED, ""));
  const unboundedLarge = stripCost(LARGE, (text) => text.replace(UNBOUNDED, ""));
  const unboundedGrowth = unboundedLarge / unboundedSmall;
  assert.equal(
    unboundedGrowth > 3.5,
    true,
    `the unbounded pattern grew ${unboundedGrowth.toFixed(2)}x, so this test is no longer measuring the difference it was written for`,
  );
  assert.equal(
    large * 10 < unboundedLarge,
    true,
    `bounded strip scanned ${large} chars against ${unboundedLarge} for the unbounded pattern; expected at least 10x fewer`,
  );
});

test("think tag tracking is linear in the reply length", () => {
  const drive = () => {
    const tracker = createThinkTagTracker();
    let seen = 0;
    return (text: string) => {
      tracker.append(text.slice(seen));
      seen = text.length;
      return tracker.endsInsideThink();
    };
  };
  const small = thinkCost(SMALL, drive);
  const large = thinkCost(LARGE, drive);
  const growth = large / small;
  assert.equal(
    growth < 2.5,
    true,
    `tracker cost grew ${growth.toFixed(2)}x for twice the reply (${small} -> ${large} chars scanned); that is not linear`,
  );

  // Each character is read a small fixed number of times, so the total is a
  // multiple of the reply, not of the reply squared.
  assert.equal(
    large < LARGE * 30,
    true,
    `tracker scanned ${large} chars for a ${LARGE} character reply`,
  );

  const rescanSmall = thinkCost(SMALL, () => hasUnclosedThinkTag);
  const rescanLarge = thinkCost(LARGE, () => hasUnclosedThinkTag);
  const rescanGrowth = rescanLarge / rescanSmall;
  assert.equal(
    rescanGrowth > 3.5,
    true,
    `the full rescan grew ${rescanGrowth.toFixed(2)}x, so this test is no longer measuring the difference it was written for`,
  );
  assert.equal(
    large * 10 < rescanLarge,
    true,
    `tracker scanned ${large} chars against ${rescanLarge} for the full rescan; expected at least 10x fewer`,
  );
});

test("the whole per-arrival path is linear in the reply length", () => {
  // The three scans above are each linear on their own; this is the one that
  // says so about the sequence the adapter actually runs, parse included. It
  // is the guard against any future arrival-time step that wants the whole
  // buffer, which is the shape every defect on this path has had.
  const cost = (chars: number): number => {
    const arrivals = arrivalsOf(buildReply(chars));
    return withScanAccounting(() => runArrivalPath(arrivals)).chars;
  };
  const small = cost(SMALL);
  const large = cost(LARGE);
  const growth = large / small;
  assert.equal(
    growth < 2.5,
    true,
    `the per-arrival path grew ${growth.toFixed(2)}x for twice the reply (${small} -> ${large} chars read); that is not linear`,
  );
  assert.equal(
    large < LARGE * 30,
    true,
    `the per-arrival path read ${large} chars for a ${LARGE} character reply`,
  );

  // And the shape it replaces, measured the same way, so the test says what it
  // is guarding against rather than only asserting a number.
  const rereadSmall = withScanAccounting(() =>
    runArrivalPathRereading(arrivalsOf(buildReply(SMALL))),
  ).chars;
  const rereadLarge = withScanAccounting(() =>
    runArrivalPathRereading(arrivalsOf(buildReply(LARGE))),
  ).chars;
  const rereadGrowth = rereadLarge / rereadSmall;
  assert.equal(
    rereadGrowth > 3.5,
    true,
    `rereading grew ${rereadGrowth.toFixed(2)}x, so this test is no longer measuring the difference it was written for`,
  );
  assert.equal(
    large * 10 < rereadLarge,
    true,
    `the per-arrival path read ${large} chars against ${rereadLarge} for rereading; expected at least 10x fewer`,
  );
});

// ------------------------------------------------------------ source pins ---

const ADAPTER = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);

/** Drop comments, so a commented-out call cannot satisfy a search. */
function withoutComments(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .split("\n")
    .map((line) => {
      const at = line.indexOf("//");
      if (at === -1) {
        return line;
      }
      // Keep a "//" inside a string literal, as in "https://".
      const quotes = line.slice(0, at).match(/["'`]/g)?.length ?? 0;
      return quotes % 2 === 1 ? line : line.slice(0, at);
    })
    .join("\n");
}

test("the adapter strips the trailing fragment through the bounded scan", () => {
  const source = withoutComments(ADAPTER);
  assert.match(source, /stripTrailingTemplatePlaceholder\(cumulativeText\)/);
  assert.doesNotMatch(
    source,
    /cumulativeText\.replace\(/,
    "the whole buffer is being rewritten again; use the bounded scan",
  );
  assert.doesNotMatch(
    source,
    /\\s\*\\\$\\\{\[\^}\]\*\\\}\\s\*\$/,
    "the unbounded pattern is back in the adapter",
  );
});

test("the reply only grows through the one call that keeps the trackers in step", () => {
  // The tracker, the placeholder watch and the incremental parse are all fed
  // the delta rather than the buffer, so each of them is only correct if it is
  // told about every character the reply gains. A second place that appended
  // to `cumulativeText` directly would leave all three describing text that no
  // longer matches, and nothing else in the file would notice.
  const source = withoutComments(ADAPTER);

  const appends = source.match(/cumulativeText\s*\+=/g) ?? [];
  assert.equal(
    appends.length,
    1,
    `the reply is appended to in ${appends.length} places; every append has to go through appendCumulative so the trackers see it`,
  );
  assert.match(
    source,
    /const appendCumulative = \(text: string\): void => \{\s*if \(!text\) \{\s*return;\s*\}\s*cumulativeText \+=/,
    "the one append site is not appendCumulative",
  );

  // Everything appendCumulative drives has to be driven from it, or the append
  // above is telling only some of them.
  for (const fed of [
    "segmentedText.appendText(text)",
    "thinkTags.append(text)",
    "placeholderWatch.append(text)",
  ]) {
    assert.equal(
      source.includes(fed),
      true,
      `appendCumulative does not feed ${fed}`,
    );
  }
});

/** Anchors and shapes this file matches against the adapter source. */
const SSE_LOOP_START = "for await (const chunk of stream) {";
const SSE_LOOP_END = "} catch (streamError) {";
const STRIP_CALL_SITE = "stripTrailingTemplatePlaceholder(cumulativeText)";
const STRIP_CALL_SITES = /stripTrailingTemplatePlaceholder\(cumulativeText\)/g;
const STRIP_GATE_HEAD =
  /if \(\s*isExternalRequest &&\s*producedReplyText &&\s*placeholderWatch\.isCandidate\(\)\s*\) \{/;
const STRIP_GATE =
  /if \(\s*isExternalRequest &&\s*producedReplyText &&\s*placeholderWatch\.isCandidate\(\)\s*\) \{\s*const stripped =\s*$/;
const STRIP_ANYWHERE = /stripTrailingTemplatePlaceholder\(/;
const FLAG_WRITES = /producedReplyText = true;/g;
const CUT_GUARD = /if \(stripped\.length !== cumulativeText\.length\) \{/;
const CUT_KEPT = /cumulativeText = stripped;/;
const FLAG_NEXT_TO_APPEND =
  /streamedChars \+= reasoning\.length \+ delta\.length;\s*producedReplyText = true;/;

/** The adapter between two anchors, without its comments. */
function regionOf(from: string, to: string, maxChars = 75_000): string {
  const start = ADAPTER.indexOf(from);
  assert.notEqual(start, -1, `"${from}" is gone; this test needs rewriting`);
  const end = ADAPTER.indexOf(to, start);
  assert.notEqual(end, -1, `"${to}" is gone; this test needs rewriting`);
  assert.ok(
    end - start < maxChars,
    `the region from "${from}" to "${to}" is ${end - start} chars; an anchor has drifted and this test needs rewriting`,
  );
  return withoutComments(ADAPTER.slice(start, end));
}

test("the arrival loop touches the reply only in ways that cannot flatten it", () => {
  // The cost on this path is not how many characters are read. `text += delta`
  // leaves a cons string, and the whole reply is copied the first time anything
  // forces it flat, which is any character access and any slice other than the
  // whole-string one. So one `charCodeAt` per arrival costs the same as a full
  // scan, and the character counting above cannot tell them apart: both read
  // the same characters. Measured over a 220,000 character reply, a per-arrival
  // `charCodeAt` is about 1000x a per-arrival append.
  //
  // Hence an allow list rather than a search for known-bad spellings: every
  // mention of the buffer inside the loop has to be one that V8 answers without
  // copying, or one that has already been gated behind a check that answers
  // from the deltas.
  const loop = regionOf(
    "for await (const chunk of stream) {",
    "} catch (streamError) {",
  );

  // Since #9098 the strip runs once on the finished reply rather than once per
  // arrival, so nothing on this path is allowed to flatten at all and there is
  // no carve-out. The strip block and its two tracker repairs are checked
  // below the loop instead, in "the trailing strip runs on the finished reply".
  assert.doesNotMatch(
    loop,
    STRIP_ANYWHERE,
    "the strip is back inside the loop, where it both flattens the reply and sees prefixes of it rather than the reply",
  );

  const allowed = [
    // `length` is stored on the cons string, so it never forces a copy.
    "cumulativeText.length",
  ];

  const mentions: string[] = [];
  for (const line of loop.split("\n")) {
    if (!line.includes("cumulativeText")) {
      continue;
    }
    if (allowed.some((form) => line.includes(form))) {
      continue;
    }
    mentions.push(line.trim());
  }
  assert.deepEqual(
    mentions,
    [],
    `these lines touch the accumulated reply on every arrival, which copies the whole reply each time:\n  ${mentions.join("\n  ")}`,
  );

  // And the loop still handles the reply, so the empty list above cannot pass
  // by the buffer having left the loop entirely.
  assert.equal(
    loop.includes("cumulativeText.length"),
    true,
    "the loop no longer reads the reply's length; this test needs rewriting",
  );
  assert.equal(
    loop.includes("appendCumulative(delta)"),
    true,
    "the loop no longer appends the delta; this test needs rewriting",
  );
});

test("nothing on the arrival path is handed the accumulated reply", () => {
  // Reading the buffer at all flattens the cons string that the appends built,
  // which costs the whole reply, so a scan that is bounded is still quadratic
  // if it is given `cumulativeText`. The three per-arrival steps therefore take
  // the delta, and the one that cannot, the strip, runs behind a check that
  // answers from the deltas.
  const source = withoutComments(ADAPTER);

  assert.doesNotMatch(
    source,
    /hasUnclosedThinkTag\(/,
    "the whole buffer is being reread; use the tracker",
  );
  for (const perArrival of [
    /thinkTags\.append\(cumulativeText\)/,
    /placeholderWatch\.append\(cumulativeText\)/,
    /segmentedText\.appendText\(cumulativeText\)/,
  ]) {
    // The seed of a resumed turn is the exception, and it is outside the loop.
    const inLoop = source
      .slice(source.indexOf("const appendCumulative"))
      .replace(/if \(cumulativeText\) \{[\s\S]*?\n {6}\}/, "");
    assert.doesNotMatch(
      inLoop,
      perArrival,
      "a per-arrival step is being handed the whole reply instead of the delta",
    );
  }

  // The strip is the one step that has to touch the buffer when it fires, so
  // it still fires only when the watch says a fragment could be there. Since
  // #9098 that is once per reply rather than once per arrival, which is why
  // the ordering below is the watch and then the strip, both outside the loop.
  assert.match(
    source,
    STRIP_GATE_HEAD,
    "the strip is running unconditionally again",
  );
  const candidate = source.indexOf("placeholderWatch.isCandidate()");
  const strip = source.indexOf(
    "stripTrailingTemplatePlaceholder(cumulativeText)",
  );
  assert.equal(candidate !== -1 && strip !== -1, true);
  assert.equal(
    candidate < strip,
    true,
    "the buffer must not be touched before the watch has been asked",
  );
});

test("the trailing strip runs on the finished reply, not on every arrival", () => {
  // #9098. The pattern is anchored at the end, so running it once per arrival
  // tested "ends with ${...}" against every PREFIX of the reply. The one
  // arrival whose buffer ended at `...${name}` was cut, and reassigning the
  // result made the cut permanent: "return `Hi, ${name}!`" arrived as
  // "return `Hi,!`". Nothing about the pattern can fix that, only where it runs.
  const source = withoutComments(ADAPTER);

  const calls = source.match(STRIP_CALL_SITES) ?? [];
  assert.equal(
    calls.length,
    1,
    "one call site: the finished reply is stripped once, and a second site would be a second chance to cut a prefix",
  );
  const strip = source.indexOf(STRIP_CALL_SITE);

  // After the loop and before the reply is turned into content, so the finished
  // text is what gets stripped and what gets saved.
  const loopEnd = source.indexOf(SSE_LOOP_END);
  assert.notEqual(loopEnd, -1);
  assert.equal(
    loopEnd < strip,
    true,
    "the strip must sit after the SSE loop, not inside it",
  );
  // Pinned as a prefix, deliberately. What this test is about is WHERE the final
  // build sits relative to the strip, not how the merge is spelled, and pinning the
  // closing parens made it fail the first time an argument was added to
  // `mergeContinuation` even though the ordering it guards was untouched. A pin that
  // breaks on unrelated edits gets "fixed" by re-pinning, and one of those days it
  // gets re-pinned past a real reordering.
  const finalBuild = source.indexOf(
    "buildAssistantContent(mergeContinuation(cumulativeText",
    strip,
  );
  assert.equal(
    finalBuild > strip,
    true,
    "the finished reply is built before the strip runs, so the strip cannot reach what is saved",
  );

  // Still external-only, and still only where this run wrote reply text of its
  // own. Local GGUF replies never leaked the fragment and must not start losing
  // template literals to a strip that stopped being gated; a continuation that
  // adds nothing holds the previous run's PARTIAL, which is a prefix, so
  // trimming its tail would be #9098 one step in.
  assert.match(
    source.slice(Math.max(0, strip - 220), strip),
    STRIP_GATE,
    "the strip is no longer gated on an external request that produced text",
  );

  // The flag has to be set where the reply grows, not somewhere a skipped
  // arrival can miss, and nowhere else.
  const flagWrites = source.match(FLAG_WRITES) ?? [];
  assert.equal(flagWrites.length, 1, "one place sets producedReplyText");
  assert.match(
    regionOf(SSE_LOOP_START, SSE_LOOP_END),
    FLAG_NEXT_TO_APPEND,
    "producedReplyText must be set next to the append, inside the loop",
  );

  // A cut still tells both trackers, and still only when it cuts.
  const stripBlock = source.slice(strip, source.indexOf("\n        }", strip));
  for (const repair of [
    "thinkTags.retract(cumulativeText)",
    "placeholderWatch.retract(cumulativeText)",
  ]) {
    assert.equal(
      stripBlock.includes(repair),
      true,
      `${repair} is not inside the strip block`,
    );
  }
  assert.match(
    stripBlock,
    CUT_GUARD,
    "the trackers are repaired when nothing was cut",
  );
  // And the cut is kept. Dropping this line leaves a strip that computes the
  // right answer and throws it away, which every other assertion here passes.
  assert.match(
    stripBlock,
    CUT_KEPT,
    "the strip's result is not assigned back, so nothing is actually removed",
  );
});
