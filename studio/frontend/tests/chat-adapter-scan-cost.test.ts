// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
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

import { readSrc } from "./helpers/kit.ts";

/** Complexity tests, not timing: they count characters each string primitive can read. */
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

test("the trailing placeholder strip is linear in the reply length", () => {
  const small = stripCost(SMALL, stripTrailingTemplatePlaceholder);
  const large = stripCost(LARGE, stripTrailingTemplatePlaceholder);
  const growth = large / small;

  // Twice the reply should double the cost; near 4x means the buffer is reread per arrival.
  assert.equal(
    growth < 2.5,
    true,
    `strip cost grew ${growth.toFixed(2)}x for twice the reply (${small} -> ${large} chars scanned); that is not linear`,
  );

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

const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");

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
  // The tracker, placeholder watch and incremental parse are fed deltas, so every append must
  // go through appendCumulative.
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
  // `text += delta` builds a cons string; any char access or partial slice flattens it, copying
  // the whole reply. Hence an allow list of buffer uses inside the loop.
  const loop = regionOf(
    "for await (const chunk of stream) {",
    "} catch (streamError) {",
  );

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
    const inLoop = source
      .slice(source.indexOf("const appendCumulative"))
      .replace(/if \(cumulativeText\) \{[\s\S]*?\n {6}\}/, "");
    assert.doesNotMatch(
      inLoop,
      perArrival,
      "a per-arrival step is being handed the whole reply instead of the delta",
    );
  }

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
  // The strip pattern is end-anchored, so running it per arrival cut text at prefixes permanently.
  const source = withoutComments(ADAPTER);

  const calls = source.match(STRIP_CALL_SITES) ?? [];
  assert.equal(
    calls.length,
    1,
    "one call site: the finished reply is stripped once, and a second site would be a second chance to cut a prefix",
  );
  const strip = source.indexOf(STRIP_CALL_SITE);

  const loopEnd = source.indexOf(SSE_LOOP_END);
  assert.notEqual(loopEnd, -1);
  assert.equal(
    loopEnd < strip,
    true,
    "the strip must sit after the SSE loop, not inside it",
  );
  // Pinned as a prefix on purpose: this guards ordering, not mergeContinuation's arguments.
  const finalBuild = source.indexOf(
    "buildAssistantContent(mergeContinuation(cumulativeText",
    strip,
  );
  assert.equal(
    finalBuild > strip,
    true,
    "the finished reply is built before the strip runs, so the strip cannot reach what is saved",
  );

  assert.match(
    source.slice(Math.max(0, strip - 220), strip),
    STRIP_GATE,
    "the strip is no longer gated on an external request that produced text",
  );

  const flagWrites = source.match(FLAG_WRITES) ?? [];
  assert.equal(flagWrites.length, 1, "one place sets producedReplyText");
  assert.match(
    regionOf(SSE_LOOP_START, SSE_LOOP_END),
    FLAG_NEXT_TO_APPEND,
    "producedReplyText must be set next to the append, inside the loop",
  );

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
  assert.match(
    stripBlock,
    CUT_KEPT,
    "the strip's result is not assigned back, so nothing is actually removed",
  );
});
