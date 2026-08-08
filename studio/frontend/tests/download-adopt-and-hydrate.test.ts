// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Two decisions the download manager makes from a single reading, both of which used to be
// wrong in the same way: treating a zero byte count as evidence of something it is not.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { carriesOverSeed, idleProbeVerdict } = await import(
  "../src/features/hub/download-manager/adopt-rules.ts"
);

test("adopting a new generation drops the previous run's byte seed", () => {
  // The persisted job describes generation 4; the backend reports 5 in flight. Seeding 4's bytes
  // while serverGeneration jumps to 5 leaves the first poll seeing no change, so the new run's
  // legitimate zero reads as "could not measure" and the card stays pinned to the old bytes.
  assert.equal(carriesOverSeed(true, 4, 5), false);
  assert.equal(carriesOverSeed(true, 4, 4), true, "the same run keeps its counters");
});

test("an unknown generation is not evidence of a new run", () => {
  // Adopt-after-reload with nothing probed: the persisted counters are all there is, and
  // discarding them would blank a card that is about to be confirmed.
  assert.equal(carriesOverSeed(true, 4, undefined), true);
  assert.equal(carriesOverSeed(true, undefined, 5), true);
  assert.equal(carriesOverSeed(true, 4, Number.NaN), true);
  assert.equal(carriesOverSeed(false, 4, 5), false, "a fresh start never seeds");
});

test("only an absent cache retires a hydrated job, not a zero reading", () => {
  // A transient measurement failure is a successful all-zero response. Calling that "gone" drops
  // a job whose partial cache is still on disk, and the user loses the resume.
  assert.equal(idleProbeVerdict(0, "/hub/models--unsloth--x"), "active");
  assert.equal(idleProbeVerdict(0, null), "gone");
  assert.equal(idleProbeVerdict(1024, null), "active", "bytes outrank a missing path");
});

test("an older backend that omits cache_path leaves the job adoptable", () => {
  // Absent is unknown, not absent-on-disk: guessing "gone" here would drop live downloads for
  // every install that has not restarted its backend yet.
  assert.equal(idleProbeVerdict(0, undefined), "active");
});
