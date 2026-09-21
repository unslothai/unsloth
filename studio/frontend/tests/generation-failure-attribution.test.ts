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
import { readFileSync } from "node:fs";
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

test("an attempt with no baseline of its own declines to attribute", () => {
  // The client never got a progress read in before its post: on the first attempt after a
  // page load, or because that read failed. A guessed baseline is the whole defect -- the
  // route only ships the counter beside a reason if it is withheld otherwise, so a run that
  // succeeded leaves the caller at 0 and ANY retained reason reads as newer than a post that
  // never arrived.
  assert.equal(
    generationFailureForAttempt({ error: REASON, generation_seq: 12 }, null),
    null,
  );
  // And a real baseline of 0 is still a baseline: nothing had run yet, so a first run's
  // failure is this attempt's.
  assert.equal(
    generationFailureForAttempt({ error: REASON, generation_seq: 1 }, 0),
    REASON,
  );
});

test("the page reads its baseline before it posts, not from whatever a poll left behind", () => {
  // Source-shape, because images-page.tsx cannot be loaded on its own: the ordering IS the
  // contract. Asserted as: the ref is cleared and a progress read awaited, both before the
  // generate loop that freezes seqBeforePost.
  const src = readFileSync(
    new URL("../src/features/images/images-page.tsx", import.meta.url),
    "utf8",
  );
  const cleared = src.indexOf("lastGenerationSeq.current = null;");
  const awaited = src.indexOf("await pollGenerateOnce();");
  const frozen = src.indexOf(
    "const seqBeforePost = lastGenerationSeq.current;",
  );
  assert.ok(cleared > 0, "the page keeps a baseline across generate clicks");
  assert.ok(
    awaited > cleared,
    "no progress read is awaited before the first post",
  );
  assert.ok(
    frozen > awaited,
    "the baseline is frozen before the read that would have observed it",
  );
});
