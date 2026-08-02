// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  isValidHubResourceId,
  validateHubResourceId,
} from "../src/components/resource-picker/hub-resource-id.ts";

test("accepts safe Hugging Face resource identifiers", () => {
  assert.deepEqual(validateHubResourceId("bert-base-uncased"), {
    ok: true,
    id: "bert-base-uncased",
  });
  assert.deepEqual(validateHubResourceId(" owner/repo_1.0 "), {
    ok: true,
    id: "owner/repo_1.0",
  });
  assert.deepEqual(validateHubResourceId("_owner/_repo_"), {
    ok: true,
    id: "_owner/_repo_",
  });
  assert.equal(
    validateHubResourceId(`${"a".repeat(96)}/${"b".repeat(96)}`).ok,
    true,
  );
  for (const value of ["owner/repo--v2", "owner/repo.git", "org/team/repo"]) {
    assert.equal(validateHubResourceId(value).ok, true, value);
  }
});

test("rejects malformed or unsafe Hugging Face resource ids", () => {
  for (const value of [
    "",
    "my dataset!",
    ".repo",
    "foo..bar",
    "owner//repo",
    "../repo",
    "owner/../repo",
    `${"a".repeat(257)}/repo`,
  ]) {
    assert.equal(validateHubResourceId(value).ok, false, value);
  }
});

test("exposes a predicate for filtering unselectable Hub results", () => {
  assert.equal(isValidHubResourceId("owner/repo"), true);
  assert.equal(isValidHubResourceId("owner/repo--cache-ambiguous"), true);
});
