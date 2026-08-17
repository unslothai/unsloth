// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  createThinkTagTracker,
  hasUnclosedThinkTag,
} from "../src/features/chat/utils/parse-assistant-content.ts";
import { stripTrailingTemplatePlaceholder } from "../src/features/chat/utils/trailing-template-placeholder.ts";

/**
 * Complexity tests, not timing tests: they count the characters the scans look
 * at, so they say the same thing on an idle laptop and a loaded CI box. Every
 * string primitive the scans use is counted by what it can read:
 *
 *   String replace / lastIndexOf / indexOf   the receiver's length
 *   String slice                             the length it produces
 *   RegExp exec / test                       the length of the subject
 *
 * `test` is included because the bounded strip walks the trailing whitespace run
 * one character at a time; each read shows up as a one-character subject, so a
 * runaway walk is counted rather than hidden.
 */
type Counted = { chars: number };

function withScanAccounting<T>(body: () => T): { result: T; chars: number } {
  const counted: Counted = { chars: 0 };
  const realReplace = String.prototype.replace;
  const realLastIndexOf = String.prototype.lastIndexOf;
  const realIndexOf = String.prototype.indexOf;
  const realSlice = String.prototype.slice;
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
  const small = thinkCost(SMALL, () => {
    const tracker = createThinkTagTracker();
    return (text: string) => tracker.update(text);
  });
  const large = thinkCost(LARGE, () => {
    const tracker = createThinkTagTracker();
    return (text: string) => tracker.update(text);
  });
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

test("the adapter asks the tracker once per arrival, not inside a condition", () => {
  const source = withoutComments(ADAPTER);
  assert.doesNotMatch(
    source,
    /hasUnclosedThinkTag\(/,
    "the whole buffer is being reread; use the tracker",
  );

  const updates = source.match(/thinkTags\.update\(/g) ?? [];
  assert.equal(
    updates.length,
    1,
    "the tracker has to see every state the buffer passes through, so exactly one call site",
  );

  // The call has to be its own statement: inside the `&&` chain below, the
  // short-circuit would skip arrivals, and a strip on a skipped arrival would
  // leave the tracker's recorded tag positions describing text that is gone.
  assert.match(
    source,
    /const textEndsInsideThink = thinkTags\.update\(cumulativeText\);/,
  );
  assert.match(source, /&&\s*!textEndsInsideThink\s*\)/);

  const strip = source.indexOf("stripTrailingTemplatePlaceholder(cumulativeText)");
  const update = source.indexOf("thinkTags.update(cumulativeText)");
  const ask = source.indexOf("!textEndsInsideThink");
  assert.equal(strip !== -1 && update !== -1 && ask !== -1, true);
  assert.equal(
    strip < update && update < ask,
    true,
    "the tracker must be updated after the strip and before the buffer is asked about",
  );
});
