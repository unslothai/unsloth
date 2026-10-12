// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A continued answer that was re-prefilled stays marked as such once the parts are joined. */

import assert from "node:assert/strict";
import test from "node:test";

import {
  CONTINUATION_RUN_CONFIG_KEY,
  readContinuationRequest,
  readPreemptRecomputed,
  recomputedModelIds,
  withRecomputedModel,
} from "../src/features/chat/utils/continuation.ts";

import { readSrc } from "./helpers/kit.ts";

const SENDS_IT = /\.\.\.readPreemptRecomputedFields\(metadata\),/;
const SEEDS_IT =
  /let sawPreemptRecompute = continuation\?\.preemptRecomputed === true;/;

test("the request carries the recompute and only a true one", () => {
  const carried = (preemptRecomputed: unknown) =>
    readContinuationRequest({
      custom: {
        [CONTINUATION_RUN_CONFIG_KEY]: {
          partial: "Half an answer",
          preemptRecomputed,
        },
      },
    })?.preemptRecomputed;
  assert.equal(carried(true), true);
  assert.equal(carried(undefined), undefined);
  assert.equal(carried("yes"), undefined);
});

test("the request keeps the model the recompute happened on", () => {
  const request = readContinuationRequest({
    custom: {
      [CONTINUATION_RUN_CONFIG_KEY]: {
        partial: "Half an answer",
        preemptRecomputed: true,
        preemptRecomputedModelIds: ["model-a"],
      },
    },
  });
  assert.deepEqual(request?.preemptRecomputedModelIds, ["model-a"]);
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  // Seeded from the request, and a recompute of the continuation's own adds its model.
  assert.match(adapter, /continuation\?\.preemptRecomputedModelIds \?\?/);
  assert.match(
    adapter,
    /sawPreemptRecompute = true;\s+preemptRecomputedOn = withRecomputedModel\(preemptRecomputedOn, params\.checkpoint\);/,
  );
});

test("a second recompute on another model keeps the first", () => {
  // Recomputed on A, continued on B and recomputed there too: both parts were re-prefilled.
  const joined = withRecomputedModel(["model-a"], "model-b");
  assert.deepEqual(joined, ["model-a", "model-b"]);
  assert.deepEqual(withRecomputedModel(joined, "model-a"), joined);
  const request = readContinuationRequest({
    custom: {
      [CONTINUATION_RUN_CONFIG_KEY]: {
        partial: "Half an answer",
        preemptRecomputed: true,
        preemptRecomputedModelIds: joined,
      },
    },
  });
  assert.deepEqual(request?.preemptRecomputedModelIds, ["model-a", "model-b"]);
});

test("a message saved with one recompute model still reads as that one", () => {
  assert.deepEqual(recomputedModelIds({ preemptRecomputedModelId: "model-a" }), ["model-a"]);
  assert.deepEqual(recomputedModelIds({ preemptRecomputedModelIds: ["a", "", 3, "a", "b"] }), [
    "a",
    "b",
  ]);
  assert.deepEqual(recomputedModelIds(undefined), []);
});

test("the metadata reader reads what the adapter writes", () => {
  assert.equal(
    readPreemptRecomputed({ custom: { preemptRecomputed: true } }),
    true,
  );
  assert.equal(readPreemptRecomputed({ custom: {} }), false);
  assert.equal(readPreemptRecomputed(undefined), false);
});

test("the continue button sends it and the adapter seeds its latch from it", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, SENDS_IT);
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, SEEDS_IT);
});

test("the recompute notice is published at once, before any more text", () => {
  // A legacy run whose tab closes before the next text keeps the last live yield, and the
  // detached abort skips the replacement yield, so the notice has to reach a yield itself.
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  const notice = adapter.indexOf('if (admissionStatus === "recomputed") {');
  assert.ok(notice > 0, "the recompute branch is gone");
  const branch = adapter.slice(notice, adapter.indexOf("continue;", notice));
  assert.match(branch, /yield \{\s+content: liveAssistantContent\(\),/);
  assert.match(branch, /custom: liveCustom\(\),/);
});
