// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A continued answer that was re-prefilled stays marked as such once the parts are joined. */

import assert from "node:assert/strict";
import test from "node:test";

import {
  CONTINUATION_RUN_CONFIG_KEY,
  readContinuationRequest,
  readPreemptRecomputed,
} from "../src/features/chat/utils/continuation.ts";

import { readSrc } from "./helpers/kit.ts";

const SENDS_IT =
  /readPreemptRecomputed\(metadata\) \? \{ preemptRecomputed: true \} : \{\}/;
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
