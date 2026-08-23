// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The delete preview is advisory: /api/hub/delete-cached re-runs every guard and refuses
// authoritatively. So the button it drives must be greyed out only for a block the preview
// positively knows, and background polling is reserved for holders that release themselves.

import assert from "node:assert/strict";
import { test } from "node:test";
import type { DeleteImpact } from "../src/features/hub/inventory/api.ts";
import {
  DELETE_BLOCK_REPOLL_CAP_MS,
  DELETE_BLOCK_REPOLL_MS,
  isDeleteBlocked,
  isUnverifiable,
  repollDelayMs,
  shouldRefreshDeleteImpactOnWake,
} from "../src/features/hub/catalog/delete-impact-state.ts";

function impact(overrides: Partial<DeleteImpact> = {}): DeleteImpact {
  return {
    repo_id: "unsloth/Qwen3-4B-GGUF",
    reclaimed_bytes: 2_600_000_000,
    retained_companions: [],
    freeable_companions: [],
    blocked_by: [],
    ...overrides,
  };
}

const block = (status_code: number, detail = "held") => ({ status_code, detail });

test("a preview that has not landed leaves Delete enabled", () => {
  assert.equal(isDeleteBlocked(null), false);
});

test("an unavailable preview retries when the dialog returns to the foreground", () => {
  assert.equal(shouldRefreshDeleteImpactOnWake(null), true);
  assert.equal(shouldRefreshDeleteImpactOnWake(impact()), false);
  assert.equal(
    shouldRefreshDeleteImpactOnWake(impact({ delete_block: block(409) })),
    true,
  );
  assert.equal(
    shouldRefreshDeleteImpactOnWake(impact({ delete_block: block(503) })),
    false,
  );
});

test("an unblocked preview leaves Delete enabled", () => {
  assert.equal(isDeleteBlocked(impact()), false);
});

test("a block the backend could not substantiate leaves Delete enabled", () => {
  const unverifiable = impact({ delete_block: block(503) });
  assert.equal(isUnverifiable(unverifiable), true);
  assert.equal(isDeleteBlocked(unverifiable), false);
});

test("a known shared-asset blocker still disables Delete when load state is unverifiable", () => {
  const mixed = impact({
    delete_block: block(503),
    blocked_by: ["unsloth/FLUX.1-dev"],
  });
  assert.equal(isUnverifiable(mixed), true);
  assert.equal(isDeleteBlocked(mixed), true);
  assert.equal(shouldRefreshDeleteImpactOnWake(mixed), true);
});

test("a known holder disables Delete, whichever kind it is", () => {
  assert.equal(isDeleteBlocked(impact({ delete_block: block(400) })), true);
  assert.equal(isDeleteBlocked(impact({ delete_block: block(409) })), true);
  assert.equal(isDeleteBlocked(impact({ blocked_by: ["unsloth/FLUX.1-dev"] })), true);
});

test("nothing to wait for is not re-polled", () => {
  // User-cleared and shared-asset blocks refresh on focus, not on a background timer.
  assert.equal(repollDelayMs(null, null), null);
  assert.equal(repollDelayMs(impact(), null), null);
  assert.equal(repollDelayMs(impact({ delete_block: block(503) }), null), null);
  assert.equal(repollDelayMs(impact({ delete_block: block(400) }), null), null);
  assert.equal(
    repollDelayMs(
      impact({
        delete_block: { ...block(409), retryable: false },
      }),
      null,
    ),
    null,
  );
  assert.equal(
    repollDelayMs(impact({ blocked_by: ["unsloth/FLUX.1-dev"] }), null),
    null,
  );
});

test("a holder that releases itself is re-polled, backing off to a cap", () => {
  const held = impact({ delete_block: block(409, "A download is writing this model.") });
  let delay = repollDelayMs(held, null);
  assert.equal(delay, DELETE_BLOCK_REPOLL_MS);

  const seen: number[] = [];
  for (let i = 0; i < 12 && delay !== null; i++) {
    seen.push(delay);
    delay = repollDelayMs(held, delay);
  }
  assert.deepEqual(seen.slice(0, 4), [2_000, 4_000, 8_000, 16_000]);
  assert.equal(seen.at(-1), DELETE_BLOCK_REPOLL_CAP_MS);
  assert.ok(seen.every((ms) => ms <= DELETE_BLOCK_REPOLL_CAP_MS));
});

test("a holder that releases stops the poll", () => {
  assert.equal(repollDelayMs(impact(), DELETE_BLOCK_REPOLL_CAP_MS), null);
});
