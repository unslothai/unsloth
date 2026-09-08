import assert from "node:assert/strict";
import test from "node:test";

import { shouldAdvanceQueuedSettingsEpoch } from "../src/features/chat/utils/queued-settings-epoch.ts";

test("no-op status reconciliation does not invalidate a pending queue", () => {
  const current = {
    checkpoint: "model",
    maxTokens: 2048,
    ggufVariant: "Q4_K_M",
  };
  assert.equal(
    shouldAdvanceQueuedSettingsEpoch(current, { ...current }),
    false,
  );
});

test("real setting changes invalidate pending queues unless tracking is suppressed", () => {
  const current = { checkpoint: "model-a", maxTokens: 2048 };
  const next = { checkpoint: "model-b", maxTokens: 2048 };
  assert.equal(shouldAdvanceQueuedSettingsEpoch(current, next), true);
  assert.equal(shouldAdvanceQueuedSettingsEpoch(current, next, false), false);
});
