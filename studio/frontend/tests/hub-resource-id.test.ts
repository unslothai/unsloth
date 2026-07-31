// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  isValidHubResourceId,
  validateHubResourceId,
} from "../src/components/resource-picker/hub-resource-id.ts";

test("accepts canonical Hugging Face resource ids", () => {
  assert.deepEqual(validateHubResourceId("bert-base-uncased"), {
    ok: true,
    id: "bert-base-uncased",
  });
  assert.deepEqual(validateHubResourceId(" owner/repo_1.0 "), {
    ok: true,
    id: "owner/repo_1.0",
  });
  assert.deepEqual(validateHubResourceId("owner/_repo_"), {
    ok: true,
    id: "owner/_repo_",
  });
  assert.equal(
    validateHubResourceId(`${"a".repeat(96)}/${"b".repeat(96)}`).ok,
    true,
  );
});

test("rejects malformed or unsafe Hugging Face resource ids", () => {
  for (const value of [
    "",
    "my dataset!",
    "datasets/foo/bar",
    ".repo",
    "repo.git",
    "foo..bar",
    "foo--bar",
    "../repo",
    "owner/../repo",
    `${"a".repeat(97)}/repo`,
  ]) {
    assert.equal(validateHubResourceId(value).ok, false, value);
  }
});

test("exposes a predicate for filtering unselectable Hub results", () => {
  assert.equal(isValidHubResourceId("owner/repo"), true);
  assert.equal(isValidHubResourceId("owner/repo--cache-ambiguous"), false);
});
