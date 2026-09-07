// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A turn paused so another chat could finish is not a failure, and must not be dressed as
 * one. The regression this guards is small and very easy to reintroduce: assistant-ui has
 * no "paused" status, so the obvious mapping is `error`, which paints MessagePrimitive's
 * red box and a Retry button over a turn that is merely waiting its turn on the KV cache.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  incompleteLabel,
  readIncompleteInfo,
  resetAutoContinue,
  restoredAssistantStatus,
  shouldAutoContinue,
} from "../src/features/chat/utils/continuation.ts";

const pausedMetadata = { custom: { incomplete: { reason: "paused" } } };

test("paused survives the metadata round trip", () => {
  // readIncompleteInfo validates against a fixed list; a reason missing from it is
  // silently dropped and the turn reloads as if it had completed normally.
  assert.deepEqual(readIncompleteInfo(pausedMetadata), { reason: "paused" });
});

test("paused never renders as an error", () => {
  const status = restoredAssistantStatus(pausedMetadata);
  assert.equal(status.type, "incomplete");
  assert.notEqual(
    (status as { reason: string }).reason,
    "error",
    "a paused turn must not get MessagePrimitive.Error's red box",
  );
});

test("paused does not claim Max Tokens was reached", () => {
  // The other non-error value. Truthful for `length` and a lie here: it would send the
  // user to raise a cap that had nothing to do with the pause.
  const status = restoredAssistantStatus(pausedMetadata);
  assert.equal((status as { reason: string }).reason, "cancelled");
});

test("the interrupted mapping is untouched", () => {
  // A cut stream IS the thing the user has to be told about; widening the non-error set
  // must not sweep it up.
  const status = restoredAssistantStatus({
    custom: { incomplete: { reason: "interrupted" } },
  });
  assert.equal((status as { reason: string }).reason, "error");
});

test("paused has its own explanation", () => {
  const label = incompleteLabel("paused");
  assert.ok(label.length > 0);
  assert.notEqual(label, incompleteLabel("cancelled"));
  assert.notEqual(label, incompleteLabel("interrupted"));
});

test("a paused turn is never resumed automatically", () => {
  // The pause is the backend rationing one KV cache and it resumes the response itself.
  // An automatic client continuation would ask for a second slot for a turn already
  // queued for one, making the oversubscription worse rather than better.
  resetAutoContinue();
  assert.equal(shouldAutoContinue("paused", "turn-1"), false);
  // Even with budget explicitly available and every fit check satisfied.
  assert.equal(
    shouldAutoContinue("paused", "turn-2", {
      limit: 3,
      fits: true,
      partialTokens: 10,
      promptTarget: 4096,
    }),
    false,
  );
});

test("auto-continue still fires for length", () => {
  // The negative above is only meaningful if the positive case still works.
  resetAutoContinue();
  assert.equal(shouldAutoContinue("length", "turn-3"), true);
});
