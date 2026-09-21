// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Which attempt a retained generation failure belongs to.
 *
 * The reason outlives its run because a caller whose POST was lost has nothing else to
 * read, which is why it cannot be taken at face value: the POST may never have arrived,
 * or the run that failed may be an earlier one or a concurrent client's. Reporting either
 * is a failure that did not happen here, and it skips the gallery probe that says what did.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  generationFailureForAttempt,
  newGenerationAttemptId,
} from "../src/features/images/lib/generation-failure.ts";

const REASON = "Image generation failed. The GPU ran out of memory.";
const MINE = "attempt-mine";

test("a failure carrying this attempt's own id is this attempt's", () => {
  assert.equal(
    generationFailureForAttempt(
      { error: REASON, generation_attempt: MINE },
      MINE,
    ),
    REASON,
  );
});

test("a failure from any other run is not, however recent", () => {
  // The two cases a monotonic counter cannot separate: a previous run of this client, and a
  // run a concurrent client started AFTER this post and failed before this waiter polled.
  // Both are "later than my baseline"; neither is this attempt.
  for (const other of ["attempt-earlier", "attempt-other-tab"]) {
    assert.equal(
      generationFailureForAttempt(
        { error: REASON, generation_attempt: other },
        MINE,
      ),
      null,
      other,
    );
  }
});

test("a reason that cannot be identified is not used", () => {
  // An older backend, or a run started by a request that carried no id. The gallery probe
  // still settles those, as it did before this field existed, rather than a guess here.
  for (const carried of [undefined, null, ""]) {
    assert.equal(
      generationFailureForAttempt(
        {
          error: REASON,
          generation_attempt: carried as string | null | undefined,
        },
        MINE,
      ),
      null,
      String(carried),
    );
  }
  // And an attempt with no id of its own cannot claim a reason either.
  assert.equal(
    generationFailureForAttempt(
      { error: REASON, generation_attempt: MINE },
      null,
    ),
    null,
  );
});

test("no failure is no failure, whatever the id says", () => {
  for (const error of [undefined, null, ""]) {
    assert.equal(
      generationFailureForAttempt(
        { error: error as string | null | undefined, generation_attempt: MINE },
        MINE,
      ),
      null,
      String(error),
    );
  }
});

test("a minted id is unique, and something the backend will accept", () => {
  // Bounded and patterned on the backend, because it comes off a request and goes back out
  // on a response: attempt_id is max_length 64 with ^[A-Za-z0-9_-]+$.
  const ids = new Set<string>();
  for (let i = 0; i < 64; i++) {
    const id = newGenerationAttemptId();
    assert.match(id, /^[A-Za-z0-9_-]+$/, id);
    assert.ok(id.length > 0 && id.length <= 64, `length ${id.length}`);
    ids.add(id);
  }
  assert.equal(
    ids.size,
    64,
    "minted ids collided, so two attempts could share a failure",
  );
});

test("an id is minted per post and sent with it", () => {
  // Source-shape, because images-page.tsx cannot be loaded on its own, and scoped to the
  // generate loop's body: an id minted anywhere else would not describe one post. Asserted
  // as: the mint, the payload field and the settle call all sit inside handleGenerate,
  // between its start and the next top-level callback.
  const src = readFileSync(
    new URL("../src/features/images/images-page.tsx", import.meta.url),
    "utf8",
  );
  const start = src.indexOf("const handleGenerate = useCallback(async () => {");
  assert.ok(start > 0, "handleGenerate was renamed");
  const end = src.indexOf("const handleGenerateWithRecall", start);
  assert.ok(end > start, "handleGenerateWithRecall was renamed");
  const body = src.slice(start, end);
  assert.ok(
    body.includes("const attemptId = newGenerationAttemptId();"),
    "the attempt id is not minted in the submit path",
  );
  assert.ok(
    body.includes("attempt_id: attemptId,"),
    "the minted id is not sent with the generate request",
  );
  assert.ok(
    body.indexOf("const attemptId") < body.indexOf("attempt_id: attemptId,"),
    "the id is sent before it is minted",
  );
  assert.ok(
    body.includes("attemptId,"),
    "the settling waiter is not given this attempt's id",
  );
});
