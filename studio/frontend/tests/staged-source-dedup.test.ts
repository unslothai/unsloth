// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  addStagedSources,
  EXPIRY_GRACE_MS,
  isExpired,
  stagedFromIntent,
} from "../src/features/rag/components/staged-source.ts";

const intent = (
  token: string,
  displayLabel: string,
  meta: { sizeBytes?: number | null; modifiedMs?: number | null } = {},
) =>
  ({
    id: token,
    kind: "attachment",
    sourceKind: "drop",
    displayLabel,
    path: {
      token,
      kind: "attachment",
      displayLabel,
      allowedOperations: ["attach"],
      expiresAtMs: 15 * 60_000,
      sizeBytes: meta.sizeBytes,
      modifiedMs: meta.modifiedMs,
    },
  }) as unknown as Parameters<typeof stagedFromIntent>[0];

// The bug: absent size/mtime both defaulted to 0, so every same-named file
// collapsed onto one signature and the second was rejected as a duplicate.
test("two same-named drops without metadata both stage", () => {
  const first = stagedFromIntent(intent("tok-a", "notes.pdf"));
  const second = stagedFromIntent(intent("tok-b", "notes.pdf"));
  const { next, duplicates } = addStagedSources([first], [second]);
  assert.equal(next.length, 2);
  assert.deepEqual(duplicates, []);
});

test("a drop missing only its mtime is still distinguishable", () => {
  const first = stagedFromIntent(intent("tok-a", "notes.pdf", { sizeBytes: 12 }));
  const second = stagedFromIntent(intent("tok-b", "notes.pdf", { sizeBytes: 12 }));
  assert.equal(addStagedSources([first], [second]).next.length, 2);
});

test("complete metadata still dedups a genuine repeat", () => {
  const meta = { sizeBytes: 12, modifiedMs: 99 };
  const first = stagedFromIntent(intent("tok-a", "notes.pdf", meta));
  const second = stagedFromIntent(intent("tok-b", "notes.pdf", meta));
  const { next, duplicates } = addStagedSources([first], [second]);
  assert.equal(next.length, 1);
  assert.deepEqual(duplicates, ["notes.pdf"]);
});

// The bug: staged native tokens outlive the native layer's TTL, so Create
// redeemed a pruned token and failed after the project already existed.
test("a token past its TTL reads as expired", () => {
  const entry = stagedFromIntent(intent("tok-a", "notes.pdf"));
  assert.equal(isExpired(entry, 15 * 60_000 + 1), true);
});

test("a token inside the grace window is already treated as gone", () => {
  const entry = stagedFromIntent(intent("tok-a", "notes.pdf"));
  assert.equal(isExpired(entry, 15 * 60_000 - EXPIRY_GRACE_MS + 1), true);
});

test("a token with time left is kept", () => {
  const entry = stagedFromIntent(intent("tok-a", "notes.pdf"));
  assert.equal(isExpired(entry, 60_000), false);
});
