// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { preferFullToolOutput, preferSanitizedFullToolOutput, toolResultText } = await import(
  "../src/features/chat/tool-output-result.ts"
);

// A timed-out python/terminal call returns the output it had already printed and then says
// it timed out. Past the model's cap that reads
// `<head>\n\n... (truncated ...)\nExecution timed out after N seconds.`, and the live stdout
// the card kept has no such sentence anywhere in it -- only the backend writes it. So
// substituting the fuller stream for the result, which is what the card does whenever the
// result is a truncated prefix of the stream, used to drop the status and leave a card that
// reads as a command that finished normally.
const SENTENCE = "Execution timed out after 300 seconds.";
const HEAD = "x".repeat(40);
const STREAM = `${"x".repeat(200)}\n`;
const NOTICE =
  "\n\n... (truncated to 40 chars for the model; 201 chars total. The full output is " +
  "not retained here; any files the code wrote persist in the working directory.)";

test("a truncated timed-out card still says the call timed out", () => {
  const card = preferFullToolOutput(STREAM, `${HEAD}${NOTICE}\n${SENTENCE}`);

  // The stream is the better copy of the output -- it is the part the cap cut -- but the
  // status it never carried has to survive with it.
  assert.ok(card.startsWith(HEAD), "the stream was not preserved");
  assert.ok(card.includes(SENTENCE), "the card no longer says the call timed out");
  // Once, not once per source: appending the whole result would repeat the output.
  assert.equal(card.split(SENTENCE).length - 1, 1);
  assert.ok(!card.includes(NOTICE.trimStart()), "the model's cap notice leaked into the card");
});

test("a completed truncated card still shows the stream alone", () => {
  // The control: with no status after the footer there is nothing to re-attach, and the
  // card must not grow a second copy of the output.
  assert.equal(preferFullToolOutput(STREAM, `${HEAD}${NOTICE}`), STREAM);
});

test("an untruncated timed-out card is left exactly as the backend wrote it", () => {
  const result = `progress\n\n${SENTENCE}`;

  assert.equal(preferFullToolOutput("progress\n", result), result);
});

test("a failed call still re-attaches its exit prefix rather than the timeout one", () => {
  // The sibling case this one is modelled on, asserted here so the new branch cannot
  // shadow it.
  assert.equal(preferFullToolOutput("boom\n", "Exit code 1:\nboom\n"), "Exit code 1:\nboom\n");
});

test("a stream whose marker line the backend indented is shown once", () => {
  // The result carries the backend's indent before a marker line; the stream does not.
  const timedOut = "progress\n__RAG_SOURCES__:[]\nfinished step\n";
  const timedOutCard = preferFullToolOutput(
    timedOut,
    `progress\n __RAG_SOURCES__:[]\nfinished step\n\n${SENTENCE}`,
  );
  assert.equal(timedOutCard.split("finished step").length - 1, 1, timedOutCard);
  assert.ok(timedOutCard.endsWith(SENTENCE), timedOutCard);

  const completed = "progress\n__FILES__:[]\nfinished step\n";
  const completedCard = preferFullToolOutput(completed, "progress\n __FILES__:[]\nfinished step\n");
  assert.equal(completedCard.split("finished step").length - 1, 1, completedCard);
});

test("a colored marker line is shown once whether or not the result was truncated", () => {
  // The backend indents only a line-start marker, so a colored one stays unindented until
  // stripping ANSI moves it to the line start.
  const esc = String.fromCharCode(27);
  const line = `${esc}[31m__FILES__:[]${esc}[0m`;

  const short = `progress\n${line}\nfinished\n`;
  const shortCard = preferSanitizedFullToolOutput(short, `${short}\n${SENTENCE}`);
  assert.equal(shortCard.split("progress").length - 1, 1, shortCard);

  const long = `progress\n${line}\n${"y".repeat(200)}\n`;
  const longCard = preferSanitizedFullToolOutput(long, `progress\n${line}\n${NOTICE}\n${SENTENCE}`);
  assert.equal(longCard.split("progress").length - 1, 1, longCard);
  assert.ok(longCard.endsWith(SENTENCE), longCard);
  assert.ok(!longCard.includes("... (truncated"), longCard);
});

test("an escape sequence the killed program left open does not swallow the timeout status", () => {
  // Stripping consumes everything after an unterminated OSC, including an appended status.
  const esc = String.fromCharCode(27);
  const stream = `progress\n${esc}]0;running`;
  const card = preferSanitizedFullToolOutput(stream, `${stream}\n${SENTENCE}`);
  assert.ok(card.endsWith(SENTENCE), JSON.stringify(card));
});

test("the timeout status survives a card built from the raw tool result", () => {
  // Stripping a string result before reconciling lost the status, with or without a saved stream.
  const esc = String.fromCharCode(27);
  const stream = `progress\n${esc}]0;running`;
  const result = `${stream}\n${SENTENCE}`;
  for (const full of [stream, ""]) {
    const card = preferSanitizedFullToolOutput(full, toolResultText(result));
    assert.ok(card.endsWith(SENTENCE), JSON.stringify(card));
  }
  assert.equal(toolResultText({ ok: true }), '{\n  "ok": true\n}');
});

test("an escape sequence the cap cut open does not repeat the output on a timed-out card", () => {
  // Stripping an escape the cap cut open also ate the footer, so the whole result was appended.
  const esc = String.fromCharCode(27);
  const bel = String.fromCharCode(7);
  const stream = `progress\n${esc}]0;${"t".repeat(200)}${bel}done with step\n`;
  const result = `progress\n${esc}]0;${"t".repeat(20)}${NOTICE}\n${SENTENCE}`;
  for (const full of [stream, ""]) {
    const card = preferSanitizedFullToolOutput(full, result);
    assert.equal(card.split("progress").length - 1, 1, JSON.stringify(card));
    assert.ok(card.endsWith(SENTENCE), JSON.stringify(card));
  }
});
