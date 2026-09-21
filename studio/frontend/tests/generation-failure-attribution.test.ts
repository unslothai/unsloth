// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Which attempt a retained generation failure belongs to.
 *
 * The reason has to outlive the run it came from, because a caller whose POST was lost past
 * the proxy window has nothing else to read. That is exactly why it cannot be taken at face
 * value: if the POST never reached the backend no run started, so the reason is a previous
 * one's, and attributing it here also skips the gallery probe that would report the truth,
 * which is that the request never arrived.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { generationFailureForAttempt } from "../src/features/images/lib/generation-failure.ts";

const REASON = "Image generation failed. The GPU ran out of memory.";

test("a failure from a run that started after this POST is this attempt's", () => {
  assert.equal(
    generationFailureForAttempt({ error: REASON, generation_seq: 8 }, 7),
    REASON,
  );
});

test("a failure from a run that started before this POST is not", () => {
  // The reported case: a previous generation failed, then this POST was lost before the
  // backend ever saw it, so no run started and the counter never moved.
  for (const seq of [7, 6, 0]) {
    assert.equal(
      generationFailureForAttempt({ error: REASON, generation_seq: seq }, 7),
      null,
      `seq=${seq}`,
    );
  }
});

test("a reason that cannot be dated is not used", () => {
  // A backend older than generation_seq. The gallery probe still settles those, as it did
  // before this field existed, rather than a guess being made here.
  for (const seq of [undefined, null]) {
    assert.equal(
      generationFailureForAttempt(
        { error: REASON, generation_seq: seq as number | null | undefined },
        0,
      ),
      null,
      String(seq),
    );
  }
});

test("no failure is no failure, however the counter reads", () => {
  for (const error of [undefined, null, ""]) {
    assert.equal(
      generationFailureForAttempt(
        { error: error as string | null | undefined, generation_seq: 99 },
        0,
      ),
      null,
      String(error),
    );
  }
});
