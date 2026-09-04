// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { stripTrailingTemplatePlaceholder } from "../src/features/chat/utils/trailing-template-placeholder.ts";

/**
 * #9098. The trailing `${...}` strip is a statement about a FINISHED reply: a
 * provider occasionally leaves a fragment on the end of an otherwise complete
 * answer. Run once per SSE arrival it was tested against every PREFIX of that
 * answer instead, and any prefix that happened to end at a complete `${...}`
 * was cut. The adapter assigns the result back, so the cut was permanent and
 * the rest of the reply streamed in on top of the hole:
 *
 *     in    return `Hi, ${name}!`      21 chars
 *     out   return `Hi,!`              13 chars
 *
 * The two modes below are the before and the after. Both call the same shipped
 * function; only WHEN it is called differs, which is the whole change. The
 * function itself is not touched, so its own suite
 * (`trailing-template-placeholder.test.ts`) still describes what it removes.
 */

/** Old: strip after every arrival, assigning the result back. */
function replayPerArrival(chunks: readonly string[]): string {
  let buffer = "";
  for (const chunk of chunks) {
    buffer += chunk;
    buffer = stripTrailingTemplatePlaceholder(buffer);
  }
  return buffer;
}

/** New: accumulate, then strip the finished reply once. */
function replayAtEnd(chunks: readonly string[]): string {
  let buffer = "";
  for (const chunk of chunks) {
    buffer += chunk;
  }
  return stripTrailingTemplatePlaceholder(buffer);
}

/** Split `text` into `size`-character arrivals. */
function chunked(text: string, size: number): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) {
    out.push(text.slice(at, at + size));
  }
  return out;
}

type Case = {
  name: string;
  /** The reply as the provider sends it, split into arrivals. */
  chunks: string[];
  /** What the user must end up with. */
  expect: string;
  /**
   * Whether the per-arrival placement lost text on this input. `true` marks a
   * case that discriminates the fix from the bug by itself; `false` marks one
   * that was already correct and must stay correct.
   */
  lostBefore: boolean;
};

const FULL = "return `Hi, ${name}!`";

const CASES: Case[] = [
  {
    // The issue, verbatim, one character per arrival: the worst case, because
    // every prefix of the reply is a buffer the strip gets to see.
    name: "the reported reproduction, one character per arrival",
    chunks: [...FULL],
    expect: FULL,
    lostBefore: true,
  },
  {
    // The case the strip exists for. Mistral magistral leaks a fragment onto
    // the end of a complete answer; deleting the strip outright brings that
    // back, so it is pinned here rather than assumed.
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
    // Two arrivals, split so the closing brace lands on the second. The strip
    // cannot match the first, which is exactly why the bug was believed
    // impossible: it fires on the arrival that COMPLETES the fragment.
    name: "a placeholder split across two arrivals",
    // The boundary lands right after the closing brace, which is the arrival
    // the strip fires on. Move it one character later and this reply survives
    // the old placement untouched: the bug is a property of where the chunks
    // fell, not of the reply.
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
    // Arrivals cut so one of them ends at the closing brace, as a real
    // token boundary can. Chunked any other way this reply comes through
    // whole, which is why the bug looks intermittent from the outside.
    chunks: ["```js\n", "const s = ", "`v=${v}", "`;\n", "```"],
    expect: "```js\nconst s = `v=${v}`;\n```",
    lostBefore: true,
  },
  {
    // #9088 landed CRLF handling on this path. `\r` is whitespace to the
    // pattern, so a CRLF reply gives the strip more places to fire, not fewer.
    name: "CRLF line endings",
    chunks: ["line one\r\n", "const t = `${q}", "`;\r\n", "line three"],
    expect: "line one\r\nconst t = `${q}`;\r\nline three",
    lostBefore: true,
  },
  {
    // Both at once: a real template literal in the body AND a leaked fragment
    // on the end. The fix has to keep one and remove the other.
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
  // Without this the suite could go green against a corpus that never
  // exercised the bug, which is the failure mode a passing test hides best.
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
  // The property behind the fix, stated directly. An SSE stream chunks
  // arbitrarily, so a placement whose answer depends on where the chunk
  // boundaries fell is wrong however good it looks on one recording.
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
  // The mirror of the test above: if the per-arrival placement also satisfied
  // it, the test would be measuring nothing.
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
  // The strip only ever takes a suffix off, so a correct placement can shorten
  // the reply and can never rewrite the middle of it. This is the sharpest
  // statement of the bug available: the old placement BREAKS it, because a cut
  // made mid-stream sits in the middle of the finished reply.
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
  // Fixed seed, so a failure is reproducible from the message alone.
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

    // The fix's guarantee: one answer, the one the whole reply deserves.
    assert.equal(
      atEnd,
      stripTrailingTemplatePlaceholder(reply),
      `atEnd disagreed on ${JSON.stringify(reply)} at size ${size}`,
    );
    // And what it returns is always a prefix of the reply. The old placement
    // is not: on 4,000 random replies it produced a non-prefix often enough to
    // be counted below, which is text removed from the middle.
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

// ---------------------------------------------------------- the shipped placement ---

const ADAPTER = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);

/**
 * Which of the two modes above the adapter is actually running.
 *
 * Read out of the source rather than assumed, so the corpus is exercised
 * through the placement that ships. Without this the cases would go on passing
 * against `replayAtEnd` no matter what the adapter did, which is the exact
 * shape of a test that measures nothing.
 */
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

// ------------------------------------------------------------- continuation ---

/**
 * A whole run, the way the adapter does it: the buffer starts at whatever a
 * Continue was seeded with, and the strip is skipped entirely unless this run
 * appended reply text of its own.
 */
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
  // A Continue run is seeded with the previous run's partial. If it finishes
  // without a text or reasoning delta, having emitted only a tool call, the
  // buffer holds nothing but that partial. The partial is the middle of a reply
  // someone is still writing, so trimming its tail is #9098 one step in: the
  // user presses Continue again and the text is already gone.
  assert.equal(replayRun(SEEDED_PARTIAL, []), SEEDED_PARTIAL);
  // And the strip would have cut it, so the gate is what saves it rather than
  // the input happening not to match.
  assert.equal(stripTrailingTemplatePlaceholder(SEEDED_PARTIAL), "greet(`Hi,");
});

test("a continuation that does add text is finished normally", () => {
  // The gate must not turn into "continuations are never trimmed". A Continue
  // that writes the rest of the answer produces a finished reply like any
  // other, artefact and all.
  assert.equal(replayRun(SEEDED_PARTIAL, ["!", "`)"]), "greet(`Hi, ${name}!`)");
  assert.equal(
    replayRun("The answer is 42.", [" ${", "answer}"]),
    "The answer is 42.",
  );
});
