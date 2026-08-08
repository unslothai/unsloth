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

test("a variant whose own files are gone does not survive on a sibling's cache dir", () => {
  // Sibling quants share one repo cache directory. Delete Q4_K_M's files while Q8_0 keeps the
  // dir alive and the reading is "zero bytes, and a directory exists" -- which read as
  // resumable, adopted a phantom, and blocked a fresh download of that same variant until the
  // idle grace expired sixty seconds later.
  assert.equal(
    idleProbeVerdict(0, "/hub/models--unsloth--x", false),
    "gone",
    "the repo dir is the wrong granularity for a variant",
  );
  // Positive evidence only in that one direction: unknown, and an older backend that never
  // sends the field, both leave the repo-level rule in charge.
  assert.equal(idleProbeVerdict(0, "/hub/models--unsloth--x", null), "active");
  assert.equal(idleProbeVerdict(0, "/hub/models--unsloth--x", undefined), "active");
  // And bytes still outrank everything: something is on disk for this target.
  assert.equal(idleProbeVerdict(4096, "/hub/models--unsloth--x", false), "active");
  assert.equal(idleProbeVerdict(0, null, true), "gone", "no cache at all is still gone");
});

test("a measured scan with no cache path retires the job however it was serialized", () => {
  // /api/hub/gguf-download-progress sets response_model_exclude_none, so its measured-empty
  // answer OMITS cache_path rather than sending null. Read as "older backend, unknown", that
  // re-adopted a job whose cache directory had been deleted and blocked a fresh download of it.
  assert.equal(idleProbeVerdict(0, undefined, null, true), "gone");
  assert.equal(idleProbeVerdict(0, null, null, true), "gone");
  // A measured scan that DID find the cache is still active, and an unmeasured one is unknown.
  assert.equal(idleProbeVerdict(0, "/hub/models--unsloth--x", null, true), "active");
  assert.equal(idleProbeVerdict(0, undefined, null, false), "active");
  // An older backend sends no flag at all and keeps the null-only rule.
  assert.equal(idleProbeVerdict(0, undefined, null, undefined), "active");
});


test("a scan that never happened does not retire a job", () => {
  // The reading is unknown, not empty. It travels as its own flag because these responses go
  // through DownloadProgressResponse, whose cache_path defaults to null -- so omitting the key
  // was reinstated as an explicit null on the wire and the distinction died before the
  // frontend saw it, on every route.
  assert.equal(idleProbeVerdict(0, null, null, false), "active");
  // It outranks the target answer too: nothing was established either way.
  assert.equal(idleProbeVerdict(0, null, false, false), "active");
  // A measured scan behaves exactly as before, and so does an older backend that omits it.
  assert.equal(idleProbeVerdict(0, null, null, true), "gone");
  assert.equal(idleProbeVerdict(0, null, null, undefined), "gone");
});
