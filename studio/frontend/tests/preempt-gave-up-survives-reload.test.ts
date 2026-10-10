// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A give-up ends on `length` because that is the shape a continuation resumes from. The live
 * adapter relabels it `paused`; a reload before the adapter settled the message went through
 * durable recovery instead, which read the run's `length` at face value and handed the turn
 * to the automatic continuation, asking for another slot in the cache that just refused one.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { generationRecoveryMetadata } = await import(
  "../src/features/chat/utils/chat-generation-recovery.ts"
);
const { shouldAutoContinue } = await import("../src/features/chat/utils/continuation.ts");

const recover = (options: { lengthLimited: boolean; preemptGaveUp?: boolean }) =>
  generationRecoveryMetadata({
    current: { generationRunId: "run-1" },
    runId: "run-1",
    status: "completed",
    cursor: 4,
    lastEventSeq: 4,
    ...options,
  });

test("a recovered give-up is paused, and paused wins over length", () => {
  assert.deepEqual(recover({ lengthLimited: true, preemptGaveUp: true }).incomplete, {
    reason: "paused",
  });
  assert.deepEqual(recover({ lengthLimited: false, preemptGaveUp: true }).incomplete, {
    reason: "paused",
  });
});

test("without the give-up a length stop recovers as length", () => {
  assert.deepEqual(recover({ lengthLimited: true }).incomplete, { reason: "length" });
  assert.equal(recover({ lengthLimited: false }).incomplete, undefined);
});

test("the recovered pause does not auto-continue", () => {
  const paused = recover({ lengthLimited: true, preemptGaveUp: true });
  const reason = (paused.incomplete as { reason: "paused" }).reason;
  assert.equal(shouldAutoContinue(reason, "run-1", { fits: true }), false);
  assert.equal(shouldAutoContinue("length", "run-1", { fits: true }), true);
});

test("the recovery publish derives the give-up the way the live adapter does", () => {
  // The rule lives in one closure, so the wiring is pinned rather than trusted.
  const source = readFileSync(
    fileURLToPath(new URL("../src/features/chat/runtime-provider.tsx", import.meta.url)),
    "utf8",
  );
  const publish = source.slice(source.indexOf("const publish = async (run: ChatGenerationRun)"));
  const call = publish.slice(0, publish.indexOf("generationRecoveryMetadata({") + 400);
  assert.match(call, /isPreemptGaveUp\(/);
  assert.match(call, /!completedAfterGivingUp\(run\.finishReason\)/);
  assert.match(call, /\bpreemptGaveUp,/);
});
