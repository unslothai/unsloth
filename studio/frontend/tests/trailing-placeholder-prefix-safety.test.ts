// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { stripTrailingTemplatePlaceholder } from "../src/features/chat/utils/trailing-template-placeholder.ts";

import { readSrc } from "./helpers/kit.ts";

/**
 * A FINISHED reply's trailing `${...}` strip, run on every streamed prefix, permanently cut
 * any prefix ending at a complete `${...}`. Old vs new placement of the same function.
 */

function replayPerArrival(chunks: readonly string[]): string {
  let buffer = "";
  for (const chunk of chunks) {
    buffer += chunk;
    buffer = stripTrailingTemplatePlaceholder(buffer);
  }
  return buffer;
}

function replayAtEnd(chunks: readonly string[]): string {
  let buffer = "";
  for (const chunk of chunks) {
    buffer += chunk;
  }
  return stripTrailingTemplatePlaceholder(buffer);
}

function chunked(text: string, size: number): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) {
    out.push(text.slice(at, at + size));
  }
  return out;
}

type Case = {
  name: string;
  chunks: string[];
  expect: string;
  /** `true` marks a case that alone discriminates the fix from the bug. */
  lostBefore: boolean;
};

const FULL = "return `Hi, ${name}!`";

const CASES: Case[] = [
  {
    name: "the reported reproduction, one character per arrival",
    chunks: [...FULL],
    expect: FULL,
    lostBefore: true,
  },
  {
    // Mistral magistral leaks a fragment onto a finished answer; this is why the strip exists.
    name: "a fragment genuinely at the end of a finished reply is still removed",
    chunks: ["The answer ", "is 42.", " ${answer}"],
    expect: "The answer is 42.",
    lostBefore: false,
  },
  {
    name: "nested placeholders",
    chunks: [..."`${a${b}}`"],
    expect: "`${a${b}}`",
    lostBefore: true,
  },
  {
    name: "several placeholders in one reply",
    chunks: [..."const s = `${x} and ${y}`;"],
    expect: "const s = `${x} and ${y}`;",
    lostBefore: true,
  },
  {
    name: "a placeholder split across two arrivals",
    chunks: ["greet(`Hi, ${na", "me}", "!`)"],
    expect: "greet(`Hi, ${name}!`)",
    lostBefore: true,
  },
  {
    name: "an unterminated ${ is never touched",
    chunks: [..."the shell wants ${HOME"],
    expect: "the shell wants ${HOME",
    lostBefore: false,
  },
  {
    name: "a placeholder inside a fenced code block",
    chunks: ["```js\n", "const s = ", "`v=${v}", "`;\n", "```"],
    expect: "```js\nconst s = `v=${v}`;\n```",
    lostBefore: true,
  },
  {
    // `\r` is whitespace to the pattern, so CRLF gives the strip more places to fire.
    name: "CRLF line endings",
    chunks: ["line one\r\n", "const t = `${q}", "`;\r\n", "line three"],
    expect: "line one\r\nconst t = `${q}`;\r\nline three",
    lostBefore: true,
  },
  {
    name: "a template literal in the body and a leaked fragment on the end",
    chunks: chunked("use `${name}` here. ${answer}", 2),
    expect: "use `${name}` here.",
    lostBefore: true,
  },
];

test("every case survives the stream intact", () => {
  for (const item of CASES) {
    assert.equal(
      replayAtEnd(item.chunks),
      item.expect,
      `${item.name}: the finished reply is wrong`,
    );
  }
});

test("the cases that discriminate really did lose text before", () => {
  // Guards against a corpus that never exercised the bug.
  for (const item of CASES) {
    const before = replayPerArrival(item.chunks);
    assert.equal(
      before !== item.expect,
      item.lostBefore,
      `${item.name}: expected lostBefore=${item.lostBefore}, but the ` +
        `per-arrival placement produced ${JSON.stringify(before)}`,
    );
  }
  assert.equal(
    CASES.filter((item) => item.lostBefore).length,
    7,
    "the corpus must keep discriminating; a case that stops losing text " +
      "before the fix stops testing the fix",
  );
});

test("the reported reproduction, character for character", () => {
  assert.equal(replayPerArrival([...FULL]), "return `Hi,!`");
  assert.equal(replayPerArrival([...FULL]).length, 13);
  assert.equal(replayAtEnd([...FULL]), FULL);
  assert.equal(replayAtEnd([...FULL]).length, 21);
});

test("the finished reply does not depend on how the stream was split", () => {
  for (const item of CASES) {
    const whole = item.chunks.join("");
    for (const size of [1, 2, 3, 5, 7, 13, 1_000]) {
      assert.equal(
        replayAtEnd(chunked(whole, size)),
        stripTrailingTemplatePlaceholder(whole),
        `${item.name}: split into ${size}-character arrivals`,
      );
    }
  }
});

test("chunk-independence is a claim the old placement fails", () => {
  const disagreements = CASES.filter((item) => {
    const whole = item.chunks.join("");
    return [1, 2, 3, 5, 7, 13].some(
      (size) =>
        replayPerArrival(chunked(whole, size)) !==
        stripTrailingTemplatePlaceholder(whole),
    );
  });
  assert.equal(
    disagreements.length,
    7,
    "the per-arrival placement is supposed to be chunking-dependent; if it " +
      "is not, this corpus no longer separates the two placements",
  );
});

test("the finished reply is always a prefix of what the model sent", () => {
  // The strip only removes a suffix, so a correct placement can never rewrite the middle.
  for (const item of CASES) {
    const whole = item.chunks.join("");
    const atEnd = replayAtEnd(item.chunks);
    assert.equal(
      whole.startsWith(atEnd),
      true,
      `${item.name}: the result ${JSON.stringify(atEnd)} is not a prefix of what the model sent`,
    );
  }
});

test("the old placement spliced the middle out, which is why text vanished", () => {
  const whole = FULL;
  const before = replayPerArrival([...whole]);
  assert.equal(before, "return `Hi,!`");
  assert.equal(
    whole.startsWith(before),
    false,
    "if the old result were merely a shortened reply this would be a trimming " +
      "bug; it is not a prefix, so characters were removed from the middle",
  );
});

test("randomised replies keep the two placements apart", () => {
  let seed = 0x9098;
  const next = () => {
    seed = (seed * 1_103_515_245 + 12_345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  const alphabet = ["a", " ", "`", "$", "{", "}", "\n", "\r", "!", "${"];

  let lost = 0;
  let spliced = 0;
  for (let trial = 0; trial < 4_000; trial += 1) {
    let reply = "";
    const length = 3 + Math.floor(next() * 25);
    for (let at = 0; at < length; at += 1) {
      reply += alphabet[Math.floor(next() * alphabet.length)];
    }
    const size = 1 + Math.floor(next() * 4);
    const atEnd = replayAtEnd(chunked(reply, size));
    const perArrival = replayPerArrival(chunked(reply, size));

    assert.equal(
      atEnd,
      stripTrailingTemplatePlaceholder(reply),
      `atEnd disagreed on ${JSON.stringify(reply)} at size ${size}`,
    );
    assert.equal(
      reply.startsWith(atEnd),
      true,
      `atEnd returned a non-prefix on ${JSON.stringify(reply)}`,
    );
    if (!reply.startsWith(perArrival)) {
      spliced += 1;
    }
    if (atEnd !== perArrival) {
      lost += 1;
    }
  }
  assert.equal(
    lost > 200,
    true,
    `only ${lost} of 4,000 random replies separated the two placements; this test is no longer measuring the difference it was written for`,
  );
  assert.equal(
    spliced > 50,
    true,
    `only ${spliced} of 4,000 random replies came back from the old placement as a non-prefix; that is the data loss this fix is about`,
  );
});

const ADAPTER = readSrc("features/chat/api/chat-adapter.ts");

/** Read from the source so the corpus runs through the placement that ships. */
function shippedReplay(): {
  replay: (chunks: readonly string[]) => string;
  loopStart: number;
  loopEnd: number;
  sites: number[];
} {
  const loopStart = ADAPTER.indexOf("for await (const chunk of stream) {");
  const loopEnd = ADAPTER.indexOf("} catch (streamError) {", loopStart);
  const sites: number[] = [];
  let at = ADAPTER.indexOf("stripTrailingTemplatePlaceholder(cumulativeText)");
  while (at !== -1) {
    sites.push(at);
    at = ADAPTER.indexOf(
      "stripTrailingTemplatePlaceholder(cumulativeText)",
      at + 1,
    );
  }
  const inLoop =
    sites.length === 1 && sites[0] > loopStart && sites[0] < loopEnd;
  return {
    replay: inLoop ? replayPerArrival : replayAtEnd,
    loopStart,
    loopEnd,
    sites,
  };
}

test("the corpus is run through the placement the adapter actually ships", () => {
  const { replay, loopStart, loopEnd, sites } = shippedReplay();
  assert.ok(
    loopStart !== -1 && loopEnd !== -1,
    "the SSE loop anchors are gone; this test needs rewriting",
  );
  assert.equal(
    sites.length,
    1,
    `expected one strip call site in the adapter, found ${sites.length}`,
  );
  for (const item of CASES) {
    assert.equal(
      replay(item.chunks),
      item.expect,
      `${item.name}: the finished reply is wrong under the shipped placement`,
    );
  }
});

/** The strip is skipped unless this run appended reply text of its own. */
function replayRun(seed: string, chunks: readonly string[]): string {
  let buffer = seed;
  let produced = false;
  for (const chunk of chunks) {
    buffer += chunk;
    produced = true;
  }
  return produced ? stripTrailingTemplatePlaceholder(buffer) : buffer;
}

const SEEDED_PARTIAL = "greet(`Hi, ${name}";

test("a continuation that adds nothing leaves the seeded partial alone", () => {
  // A Continue run that emits only a tool call holds just the seeded partial, the middle of a
  // reply; trimming its tail would lose text.
  assert.equal(replayRun(SEEDED_PARTIAL, []), SEEDED_PARTIAL);
  assert.equal(stripTrailingTemplatePlaceholder(SEEDED_PARTIAL), "greet(`Hi,");
});

test("a continuation that does add text is finished normally", () => {
  assert.equal(replayRun(SEEDED_PARTIAL, ["!", "`)"]), "greet(`Hi, ${name}!`)");
  assert.equal(
    replayRun("The answer is 42.", [" ${", "answer}"]),
    "The answer is 42.",
  );
});
