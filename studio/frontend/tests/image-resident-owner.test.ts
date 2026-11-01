// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { DiffusionStatus } from "../src/features/images/api.ts";
import {
  HELD_BY_OTHER_ACCOUNT_NOTICE,
  heldByOtherAccount,
  ownDiffusionStatus,
} from "../src/features/images/lib/resident-owner.ts";
import { readSrc } from "./helpers/kit.ts";

// What /api/inference/images/status answers for another account's resident.
const hidden = { loaded: true, yours: false } as unknown as DiffusionStatus;

test("another account's resident reads as no model loaded on the Images page (#16178)", () => {
  const own = ownDiffusionStatus(hidden);
  assert.equal(own.loaded, false);
  assert.equal(heldByOtherAccount(own), true);
});

test("this account's own status passes through unchanged", () => {
  const mine = { loaded: true, yours: true, repo_id: "org/model" } as unknown as DiffusionStatus;
  assert.equal(ownDiffusionStatus(mine), mine);
  assert.equal(heldByOtherAccount(mine), false);
  const older = { loaded: true, repo_id: "org/model" } as unknown as DiffusionStatus;
  assert.equal(ownDiffusionStatus(older), older);
  assert.equal(heldByOtherAccount(older), false);
  assert.equal(heldByOtherAccount(null), false);
});

test("every status read on the Images page goes through ownDiffusionStatus", () => {
  const page = readSrc("features/images/images-page.tsx");
  const reads = page.match(/await getDiffusionStatus\(\)/g) ?? [];
  const owned = page.match(/ownDiffusionStatus\(await getDiffusionStatus\(\)\)/g) ?? [];
  assert.ok(reads.length > 0);
  assert.equal(owned.length, reads.length);
});

test("the Images page explains the account mismatch", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /heldByOtherAccount\(status\)/);
  assert.match(page, /\{HELD_BY_OTHER_ACCOUNT_NOTICE\}/);
  assert.match(HELD_BY_OTHER_ACCOUNT_NOTICE, /Another account/);
});
