// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  type ContextTruncation,
  historyCannotHelp,
  latestTurnIsTheProblem,
  latestTurnOwnTokens,
  mergeContextTruncation,
} from "../src/features/chat/utils/context-truncation.ts";

function refusal(extra: Partial<ContextTruncation>): ContextTruncation {
  return {
    dropped_messages: 0,
    fits: false,
    context_length: 4096,
    prompt_target: 3072,
    ...extra,
  };
}

// Emitted verbatim by `fit_rolling_context`, measured with the real llama.cpp tokenizer
// (b10360, gemma-4 vocab) through the bundled `gemma-4.jinja`: 4,096-token window, system
// prompt, six evictable turns, a 6,113-token MCP catalogue, and a last message of
// `{"role":"user","content":"hi"}` costing 6 rendered tokens.
const MCP_CATALOGUE_4096: ContextTruncation = {
  dropped_messages: 0,
  fits: false,
  prompt_tokens_before: 8237,
  prompt_tokens_after: 8237,
  irreducible_tokens: 6323,
  latest_turn_tokens: 6128,
  latest_turn_role: "user",
  shared_prompt_tokens: 6122,
  latest_turn_exact: true,
  context_length: 4096,
  prompt_target: 3072,
};

test("the tool catalogue is taken off the turn before the turn is blamed", () => {
  // Both counts price a whole rendered prompt and the catalogue does not cancel: 6,122 of
  // the turn's 6,128 tokens are tools the user did not send. Before the fix the toast read
  // "This message is 6,128 tokens on its own, against the 3,072 tokens this 4,096-token
  // window leaves for the prompt", about the word "hi".
  assert.equal(latestTurnOwnTokens(MCP_CATALOGUE_4096), 6);
  assert.equal(latestTurnIsTheProblem(MCP_CATALOGUE_4096, 3072), false);
});

test("the built-in catalogue alone is diagnosed the same way", () => {
  // Default install, measured the same way: 988 tokens of built-in tools, same six-token
  // "hi". Never crossed the budget even before the fix, but the toast printed 1,003.
  const builtin: ContextTruncation = {
    dropped_messages: 0,
    fits: false,
    prompt_tokens_before: 5512,
    prompt_tokens_after: 5512,
    irreducible_tokens: 3598,
    latest_turn_tokens: 1003,
    latest_turn_role: "user",
    shared_prompt_tokens: 997,
    latest_turn_exact: true,
    context_length: 4096,
    prompt_target: 3072,
  };
  assert.equal(latestTurnOwnTokens(builtin), 6);
  assert.equal(latestTurnIsTheProblem(builtin, 3072), false);
});

test("the catalogue does not cancel at any catalogue size", () => {
  // Measured at 7, 20, 200 and 2000 tools against the same thread: the floor tracks the
  // catalogue, the turn stays 6 tokens, and the raw ratio climbs from 0.279 to 0.991 while
  // the real one never moves. The verdict must not flip on how many tools are advertised.
  const measured: Array<[number, number, number]> = [
    // [catalogue floor, latest_turn_tokens, irreducible_tokens]
    [997, 1003, 3598],
    [2951, 2957, 5552],
    [29166, 29172, 31767],
    [290937, 290943, 293538],
  ];
  for (const [floor, latest, irreducible] of measured) {
    const turn = refusal({
      irreducible_tokens: irreducible,
      latest_turn_tokens: latest,
      shared_prompt_tokens: floor,
      latest_turn_role: "user",
      latest_turn_exact: true,
    });
    assert.equal(latestTurnOwnTokens(turn), 6, `floor ${floor}`);
    assert.equal(latestTurnIsTheProblem(turn, 3072), false, `floor ${floor}`);
  }
});

test("a turn that really is too big is still blamed once the floor is off", () => {
  // The fix must not silence the case the diagnosis exists for: 9,000 tokens of pasted
  // text beside the measured 997-token built-in floor, where halving it really would fit.
  const hugeTurn = refusal({
    irreducible_tokens: 10100,
    latest_turn_tokens: 9997,
    shared_prompt_tokens: 997,
    latest_turn_role: "user",
    latest_turn_exact: true,
  });
  assert.equal(latestTurnOwnTokens(hugeTurn), 9000);
  assert.equal(latestTurnIsTheProblem(hugeTurn, 3072), true);
});

test("a server that sends no floor behaves exactly as it did before the field", () => {
  // A newer client against a server predating `shared_prompt_tokens` must not subtract a
  // floor it was never told about, and must not change a number it prints.
  const oldServer = refusal({
    irreducible_tokens: 5050,
    latest_turn_tokens: 5000,
    latest_turn_role: "user",
  });
  assert.equal(latestTurnOwnTokens(oldServer), 5000);
  assert.equal(latestTurnIsTheProblem(oldServer, 3072), true);
  assert.equal(latestTurnIsTheProblem(oldServer, 8192), false);
});

test("a floor of zero is the same as no floor at all", () => {
  // The backend sends 0 for an estimated turn: that estimate prices the message's own
  // JSON and no catalogue, so it has no floor to remove.
  const estimated = refusal({
    irreducible_tokens: 5050,
    latest_turn_tokens: 5000,
    shared_prompt_tokens: 0,
    latest_turn_role: "tool",
  });
  assert.equal(latestTurnOwnTokens(estimated), 5000);
});

test("a floor can never eat the whole turn, however wrong it arrives", () => {
  // Reporting a turn as zero tokens is a worse lie than reporting the catalogue's size,
  // and a negative one prints a minus sign at the user.
  for (const bad of [5000, 5001, 999999]) {
    const turn = refusal({ latest_turn_tokens: 5000, shared_prompt_tokens: bad });
    assert.equal(latestTurnOwnTokens(turn), 1, `floor ${bad}`);
  }
  // `toLocaleString` renders NaN, Infinity and fractions straight at the user.
  for (const bad of [
    Number.NaN,
    Number.POSITIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    -1,
    12.7,
    undefined,
  ]) {
    const own = latestTurnOwnTokens(
      refusal({ latest_turn_tokens: 5000, shared_prompt_tokens: bad }),
    );
    assert.ok(Number.isInteger(own), `floor ${String(bad)} produced ${own}`);
    assert.ok(own >= 1 && own <= 5000, `floor ${String(bad)} produced ${own}`);
  }
  // And a missing turn count stays zero rather than going negative through the clamp.
  assert.equal(latestTurnOwnTokens(refusal({ shared_prompt_tokens: 6000 })), 0);
  assert.equal(latestTurnOwnTokens(undefined), 0);
  assert.equal(latestTurnOwnTokens(null), 0);
});

test("no diagnosis at all blames nothing", () => {
  assert.equal(latestTurnIsTheProblem(null, 3072), false);
  assert.equal(latestTurnIsTheProblem(undefined, 3072), false);
});

test("the estimate flag still gates the claim, after the floor is off", () => {
  // The two guards are independent. `latest_turn_exact: false` is now only the last-resort
  // branch where nothing could price the turn (an unrenderable turn is priced by
  // difference and reported exact), and that estimate does not share units with
  // `irreducible_tokens`, so it must never be quoted as the turn's size however the
  // subtraction comes out. Measured on the bundled gemma-4 template: 16,400 characters of
  // newline and tab runs estimate 8,207 tokens against 557 rendered.
  const estimatedTurn = refusal({
    irreducible_tokens: 4449,
    latest_turn_tokens: 8207,
    shared_prompt_tokens: 0,
    latest_turn_role: "tool",
    latest_turn_exact: false,
  });
  assert.equal(latestTurnIsTheProblem(estimatedTurn, 3072), false);
  // Same payload, counted rather than guessed: now it is a claim we can make.
  assert.equal(
    latestTurnIsTheProblem({ ...estimatedTurn, latest_turn_exact: true }, 3072),
    true,
  );
});

test("the floor is dropped once a later fit succeeds", () => {
  // The tool loop refits per iteration, and a floor left behind from a failed fit would be
  // subtracted from a later fit's count, moving the blame instead of removing it.
  const failed = mergeContextTruncation(undefined, {
    dropped_messages: 0,
    fits: false,
    context_length: 4096,
    irreducible_tokens: 6100,
    latest_turn_tokens: 6020,
    shared_prompt_tokens: 6000,
  });
  assert.equal(failed.shared_prompt_tokens, 6000);

  const recovered = mergeContextTruncation(failed, {
    dropped_messages: 12,
    fits: true,
    context_length: 4096,
  });
  assert.ok(!("shared_prompt_tokens" in recovered));
  assert.ok(!("latest_turn_tokens" in recovered));
});

test("a prompt whose floor is already over the window is never sent to a new chat", () => {
  // The case has to land somewhere once the turn is no longer blamed: what survives
  // eviction is a measured 6,323 tokens against a 4,096 window, so a new chat renders the
  // same catalogue and fails identically.
  assert.equal(latestTurnIsTheProblem(MCP_CATALOGUE_4096, 3072), false);
  assert.equal(historyCannotHelp(MCP_CATALOGUE_4096), true);

  // Same counts under an 8,192-token window: the floor fits, so shortening is honest
  // advice again. The window picks the wording, not the ratio.
  assert.equal(
    historyCannotHelp({
      ...MCP_CATALOGUE_4096,
      context_length: 8192,
      prompt_target: 6144,
    }),
    false,
  );
  // Below the window shortening can work: the fit refuses at `prompt_target` but passes
  // the untrimmed messages on, and llama-server serves anything under the window.
  assert.equal(
    historyCannotHelp({ ...MCP_CATALOGUE_4096, irreducible_tokens: 4095 }),
    false,
  );
  // Exactly at it is refused too, so `>=` and not `>`.
  assert.equal(
    historyCannotHelp({ ...MCP_CATALOGUE_4096, irreducible_tokens: 4096 }),
    true,
  );
  // A payload missing either number cannot make the claim.
  assert.equal(
    historyCannotHelp({ dropped_messages: 0, fits: false, irreducible_tokens: 6323 }),
    false,
  );
  assert.equal(
    historyCannotHelp({ dropped_messages: 0, fits: false, context_length: 4096 }),
    false,
  );
  assert.equal(historyCannotHelp(null), false);
  assert.equal(historyCannotHelp(undefined), false);
});

test("the third toast branch names the levers that can actually work", () => {
  const source = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  // The band moved out of "this message is too long" must not fall through to "start a
  // new chat", the one action that provably cannot work here.
  assert.match(source, /historyCannotHelp\(irreducible\)/);
  assert.match(
    source,
    /Even with every earlier turn dropped, this prompt would still be/,
  );
  assert.match(
    source,
    /the system prompt and any \" \+\n\s*\"tools that are enabled\./,
  );
});

test("the toast quotes the turn's own size, never the count that carries the floor", () => {
  const source = readFileSync(
    new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
    "utf8",
  );
  // Printing `latest_turn_tokens` directly is the defect this guards against coming back.
  assert.match(
    source,
    /\$\{latestTurnOwnTokens\(irreducible\)\.toLocaleString\(\)\} tokens on its own/,
  );
  assert.doesNotMatch(
    source,
    /latest_turn_tokens\?\.toLocaleString\(\)\} tokens on its own/,
  );
});
