// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Phase 4: every row of the heard-cut rules table, the midpoint rule, a clause-split sentence,
// and base on the run-flicker path.

import assert from "node:assert/strict";
import test from "node:test";

import {
  applyHeardCut,
  heardCut,
  heardTail,
  joinTextParts,
  laterCut,
  savedHeardCut,
  type PlayedChunk,
} from "../src/features/chat/voice/heard-cut.ts";
import { splitSpoken } from "../src/features/chat/voice/speech-text.ts";

const REPLY = "Paris is the capital. It sits on the Seine. Its museums are famous.";
const S = splitSpoken(REPLY);
const end = (i: number) => S[i].end;
// One clip per sentence, one second each, starting at t=10.
const LOG: PlayedChunk[] = S.map((s, i) => ({
  entries: [{ source: i, from: s.start, to: s.end, chars: s.text.length }],
  start: 10 + i,
  duration: 1,
}));

test("a sentence is heard once the position passes the midpoint of the piece that ends it", () => {
  assert.deepEqual(heardCut(S, LOG, 10.49, 0), { offset: 0 });
  assert.deepEqual(heardCut(S, LOG, 10.5, 0), { offset: end(0) });
  assert.deepEqual(heardCut(S, LOG, 11.6, 0), { offset: end(1) });
  assert.deepEqual(heardCut(S, LOG, 99, 0), { offset: end(2) });
});

test("a chunk's time is shared among its pieces by characters", () => {
  const both: PlayedChunk = {
    entries: [
      { source: 0, from: S[0].start, to: S[0].end, chars: 10 },
      { source: 1, from: S[1].start, to: S[1].end, chars: 30 },
    ],
    start: 0,
    duration: 4,
  };
  // Piece 0 spans 0-1 s (midpoint 0.5), piece 1 spans 1-4 s (midpoint 2.5).
  assert.deepEqual(heardCut(S, [both], 0.6, 0), { offset: end(0) });
  assert.deepEqual(heardCut(S, [both], 2.4, 0), { offset: end(0) });
  assert.deepEqual(heardCut(S, [both], 2.5, 0), { offset: end(1) });
});

test("a clause-split sentence is heard only once its remainder has mostly played", () => {
  const head = "Paris is".length;
  const log: PlayedChunk[] = [
    { entries: [{ source: 0, from: 0, to: head, chars: head }], start: 0, duration: 1 },
    { entries: [{ source: 0, from: head + 1, to: end(0), chars: 12 }], start: 1, duration: 1 },
  ];
  assert.deepEqual(heardCut(S, log, 0.9, 0), { offset: 0 }, "the head alone is not the sentence");
  assert.deepEqual(heardCut(S, log, 1.4, 0), { offset: 0 });
  assert.deepEqual(heardCut(S, log, 1.5, 0), { offset: end(0) });
});

test("a barge-in during the filler, before any reply sentence was heard, cuts at 0", () => {
  const filler: PlayedChunk = { entries: [], start: 0, duration: 1.2 };
  assert.deepEqual(heardCut(S, [filler], 0.8, 0), { offset: 0 });
  assert.deepEqual(heardCut(S, [filler, LOG[0]], 10.2, 0), { offset: 0 });
});

test("no log (the chat model spoke the reply, or read-aloud) is no cut", () => {
  assert.equal(heardCut(S, null, 99, 0), null);
});

test("on the run-flicker path the stream's offsets start at base", () => {
  const before = "Paris is the capital. ";
  const added = "It sits on the Seine. Its museums are famous.";
  const full = splitSpoken(before + added);
  const own = splitSpoken(added);
  const log: PlayedChunk[] = own.map((s, i) => ({
    entries: [{ source: i, from: s.start, to: s.end, chars: s.text.length }],
    start: i,
    duration: 1,
  }));
  assert.deepEqual(heardCut(full, log, 0.6, before.length), { offset: full[1].end });
  assert.deepEqual(heardCut(full, log, 0.4, before.length), { offset: 0 });
});

test("a second barge-in on the same reply keeps the larger offset", () => {
  assert.deepEqual(laterCut({ offset: 20 }, { offset: 8 }), { offset: 20 });
  assert.deepEqual(laterCut({ offset: 8 }, { offset: 20 }), { offset: 20 });
  assert.deepEqual(laterCut(null, { offset: 3 }), { offset: 3 });
  assert.equal(laterCut(null, null), null);
});

const tool = { type: "tool-call", toolName: "search" };
const reasoning = { type: "reasoning", text: "thinking. about it." };

test("sentences up to the cut are kept with everything between them; after the cut is hidden", () => {
  const text = "Run this:\n```\npip install x\n```\nThen restart. Then wait.";
  const sentences = splitSpoken(text);
  const restart = sentences.find((s) => s.text === "Then restart.")!;
  const out = applyHeardCut({ content: [{ type: "text", text }] }, { offset: restart.end });
  assert.equal(out.content[0].text, text.slice(0, restart.end));
  assert.match(out.content[0].text!, /pip install x/, "the code it showed stays");
  assert.doesNotMatch(out.content[0].text!, /Then wait/);
});

test("parts before the cut's part are kept, parts after it are hidden", () => {
  const content = [reasoning, { type: "text", text: "Let me check." }, tool, { type: "text", text: "It is sunny. Take a hat." }, tool];
  const { text, partStarts } = joinTextParts(content);
  assert.equal(text, "Let me check.\nIt is sunny. Take a hat.");
  assert.equal(partStarts[3], "Let me check.\n".length);
  const sunny = splitSpoken(text).find((s) => s.text === "It is sunny.")!;
  const out = applyHeardCut({ content }, { offset: sunny.end });
  assert.deepEqual(out.content, [reasoning, content[1], tool, { type: "text", text: "It is sunny." }]);
  // Cut in the first part: the tool call and everything after it never reached the user.
  const first = applyHeardCut({ content }, { offset: "Let me check.".length });
  assert.deepEqual(first.content, [reasoning, content[1]]);
  // Joined with "\n": "check" + tool + "It" is not "checkIt".
  assert.deepEqual(splitSpoken(text).map((s) => s.text), ["Let me check.", "It is sunny.", "Take a hat."]);
});

test("a cut at 0 empties the reply text but keeps what came before it", () => {
  const out = applyHeardCut({ content: [reasoning, { type: "text", text: REPLY }] }, { offset: 0 });
  assert.deepEqual(out.content, [reasoning, { type: "text", text: "" }]);
});

test("a cut is applied the same way twice, and a cut at the end changes nothing", () => {
  const message = { id: "m", content: [{ type: "text", text: REPLY }] };
  const once = applyHeardCut(message, { offset: end(1) });
  assert.deepEqual(applyHeardCut(once, { offset: end(1) }), once);
  assert.equal(applyHeardCut(message, { offset: REPLY.length }), message);
  assert.equal(applyHeardCut({ content: [tool] }, { offset: 0 }).content[0], tool);
});

test("the saved cut is read from metadata.custom.voiceHeard", () => {
  assert.deepEqual(savedHeardCut({ custom: { voiceHeard: { offset: 12, at: 1 } } }), { offset: 12 });
  for (const bad of [undefined, null, {}, { custom: {} }, { custom: { voiceHeard: { offset: -1 } } }, { custom: { voiceHeard: { offset: "3" } } }]) {
    assert.equal(savedHeardCut(bad), null, JSON.stringify(bad));
  }
});

test("the note quotes the last words heard, or none", () => {
  const content = [{ type: "text", text: "It is **sunny** in Paris today. Take a hat." }];
  const cut = { offset: "It is **sunny** in Paris today.".length };
  assert.equal(heardTail(content, cut), "It is sunny in Paris today.");
  assert.equal(heardTail(content, cut, 3), "in Paris today.");
  assert.equal(heardTail(content, { offset: 0 }), "");
});

test("the note's tail stays short for a script with no spaces between words", () => {
  const text = "東京は日本の首都で、人口が最も多い都市であり、多くの企業の本社が集まっている";
  const reply = [{ type: "text", text: text.repeat(4) }];
  const tail = heardTail(reply, { offset: reply[0].text.length });
  assert.ok(tail.length <= 80 && reply[0].text.endsWith(tail));
  assert.equal(heardTail([{ type: "text", text: "One two three four five six seven." }], { offset: 34 }), "two three four five six seven.");
  // An astral character (CJK extension B) counts as one: no half pair starts the note.
  const astral = "𠮷".repeat(100);
  assert.equal(heardTail([{ type: "text", text: astral }], { offset: astral.length }), "𠮷".repeat(80));
  // Thai vowel marks ride on their consonant: a cut by code points left one hanging at the start.
  const thai = "กิ".repeat(100) + "ก";
  assert.equal(heardTail([{ type: "text", text: thai }], { offset: thai.length }), "กิ".repeat(79) + "ก");
});

test("the note never shows an inline player's base64", () => {
  const tag = '<audio-player src="data:audio/wav;base64,UklGRiQAAABXQVZFZm10" />';
  const text = `${tag}\nIt is sunny. Take a hat.`;
  const tail = heardTail([{ type: "text", text }], { offset: text.indexOf("Take") - 1 });
  assert.equal(tail, "It is sunny.");
});
