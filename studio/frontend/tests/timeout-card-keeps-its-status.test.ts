// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { preferFullToolOutput } = await import(
  "../src/features/chat/tool-output-scope.ts"
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
