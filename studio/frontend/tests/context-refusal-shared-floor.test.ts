// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  type ContextTruncation,
  historyCannotHelp,
  latestTurnIsTheProblem,
  latestTurnOwnTokens,
  mergeContextTruncation,
} from "../src/features/chat/utils/context-truncation.ts";

import { readSrc } from "./helpers/kit.ts";

const CHAT_ADAPTER = readSrc("features/chat/api/chat-adapter.ts");

function refusal(extra: Partial<ContextTruncation>): ContextTruncation {
  return {
    dropped_messages: 0,
    fits: false,
    context_length: 4096,
    prompt_target: 3072,
    ...extra,
  };
}

// Emitted by fit_rolling_context with the real gemma-4 tokenizer and template.
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
  assert.equal(latestTurnOwnTokens(MCP_CATALOGUE_4096), 6);
  assert.equal(latestTurnIsTheProblem(MCP_CATALOGUE_4096, 3072), false);
});

test("the built-in catalogue alone is diagnosed the same way", () => {
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
  const measured: Array<[number, number, number]> = [
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
  const estimated = refusal({
    irreducible_tokens: 5050,
    latest_turn_tokens: 5000,
    shared_prompt_tokens: 0,
    latest_turn_role: "tool",
  });
  assert.equal(latestTurnOwnTokens(estimated), 5000);
});

test("a floor can never eat the whole turn, however wrong it arrives", () => {
  for (const bad of [5000, 5001, 999999]) {
    const turn = refusal({ latest_turn_tokens: 5000, shared_prompt_tokens: bad });
    assert.equal(latestTurnOwnTokens(turn), 1, `floor ${bad}`);
  }
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
  assert.equal(latestTurnOwnTokens(refusal({ shared_prompt_tokens: 6000 })), 0);
  assert.equal(latestTurnOwnTokens(undefined), 0);
  assert.equal(latestTurnOwnTokens(null), 0);
});

test("no diagnosis at all blames nothing", () => {
  assert.equal(latestTurnIsTheProblem(null, 3072), false);
  assert.equal(latestTurnIsTheProblem(undefined, 3072), false);
});

test("the estimate flag still gates the claim, after the floor is off", () => {
  // An estimated turn does not share units with irreducible_tokens, so never quote it.
  const estimatedTurn = refusal({
    irreducible_tokens: 4449,
    latest_turn_tokens: 8207,
    shared_prompt_tokens: 0,
    latest_turn_role: "tool",
    latest_turn_exact: false,
  });
  assert.equal(latestTurnIsTheProblem(estimatedTurn, 3072), false);
  assert.equal(
    latestTurnIsTheProblem({ ...estimatedTurn, latest_turn_exact: true }, 3072),
    true,
  );
});

test("the floor is dropped once a later fit succeeds", () => {
  // A floor left from a failed fit would be subtracted from a later fit's count.
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
  assert.equal(latestTurnIsTheProblem(MCP_CATALOGUE_4096, 3072), false);
  assert.equal(historyCannotHelp(MCP_CATALOGUE_4096), true);

  assert.equal(
    historyCannotHelp({
      ...MCP_CATALOGUE_4096,
      context_length: 8192,
      prompt_target: 6144,
    }),
    false,
  );
  assert.equal(
    historyCannotHelp({ ...MCP_CATALOGUE_4096, irreducible_tokens: 4095 }),
    false,
  );
  assert.equal(
    historyCannotHelp({ ...MCP_CATALOGUE_4096, irreducible_tokens: 4096 }),
    true,
  );
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
  assert.match(CHAT_ADAPTER, /historyCannotHelp\(irreducible\)/);
  assert.match(
    CHAT_ADAPTER,
    /Even with every earlier turn dropped, this prompt would still be/,
  );
  assert.match(
    CHAT_ADAPTER,
    /the system prompt and any \" \+\n\s*\"tools that are enabled\./,
  );
});

test("the toast quotes the turn's own size, never the count that carries the floor", () => {
  assert.match(
    CHAT_ADAPTER,
    /\$\{latestTurnOwnTokens\(irreducible\)\.toLocaleString\(\)\} tokens on its own/,
  );
  assert.doesNotMatch(
    CHAT_ADAPTER,
    /latest_turn_tokens\?\.toLocaleString\(\)\} tokens on its own/,
  );
});
