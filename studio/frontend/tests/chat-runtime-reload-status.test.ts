// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { readIncompleteInfo, restoredAssistantStatus } = await import(
  "../src/features/chat/utils/continuation.ts"
);
const {
  generationNeedsRecovery,
  generationRecoveryMetadata,
  shouldPreserveGenerationMetadata,
  subscribeGenerationRecoveryTriggers,
} = await import("../src/features/chat/utils/chat-generation-recovery.ts");

test("stored assistant status remains truthful after reload", () => {
  for (const reason of ["interrupted", "length"] as const) {
    const metadata = { custom: { incomplete: { reason } } };
    assert.deepEqual(restoredAssistantStatus(metadata), {
      type: "incomplete",
      reason: "error",
    });
    assert.deepEqual(readIncompleteInfo(metadata), { reason });
  }

  assert.deepEqual(restoredAssistantStatus({ custom: {} }), {
    type: "complete",
    reason: "unknown",
  });
  assert.deepEqual(restoredAssistantStatus(undefined), {
    type: "complete",
    reason: "unknown",
  });
});

test("reload, wake, and stale-tab recovery stays monotonic and truthful", () => {
  const apply = (status: "running" | "completed" | "failed" | "cancelled", cursor: number, lengthLimited = false) => generationRecoveryMetadata({ current: { generationRunId: "run-1" }, runId: "run-1", status, cursor, lastEventSeq: 4, lengthLimited });
  assert.deepEqual(
    [["running", 2], ["completed", 2, true], ["completed", 4, true], ["failed", 4], ["cancelled", 4]].map(([status, cursor, limited]) => {
      const metadata = apply(status as "running" | "completed" | "failed" | "cancelled", cursor as number, Boolean(limited));
      return [generationNeedsRecovery(metadata), metadata.incomplete];
    }),
    [[true, { reason: "cancelled" }], [true, { reason: "cancelled" }], [false, { reason: "length" }], [false, { reason: "interrupted" }], [false, { reason: "cancelled" }]],
  );

  const windowTarget = new EventTarget(), documentTarget = Object.assign(new EventTarget(), { visibilityState: "hidden" });
  let recoveries = 0;
  const unsubscribe = subscribeGenerationRecoveryTriggers(windowTarget, documentTarget, () => { recoveries += 1; });
  windowTarget.dispatchEvent(new Event("online"));
  windowTarget.dispatchEvent(new Event("pageshow"));
  documentTarget.dispatchEvent(new Event("visibilitychange"));
  documentTarget.visibilityState = "visible";
  documentTarget.dispatchEvent(new Event("visibilitychange"));
  unsubscribe();
  windowTarget.dispatchEvent(new Event("online"));
  assert.equal(recoveries, 3);

  const existing = { generationRunId: "run-1", generationSeq: 4, generationStatus: "completed", generationSettled: true, serverManaged: true };
  const incoming = { ...existing };
  assert.deepEqual([incoming, { ...incoming, generationSeq: 3 }, { ...incoming, generationRunId: "run-2" }, { ...incoming, generationStatus: "running" }, { ...incoming, generationSettled: false }].map((candidate) => shouldPreserveGenerationMetadata(existing, candidate)), [false, true, true, true, true]);
});
