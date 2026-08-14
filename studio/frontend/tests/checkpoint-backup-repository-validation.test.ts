// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { isRepositoryId } from "../src/features/studio/sections/repository-id.ts";

test("repository IDs require exactly an owner and repository", () => {
  assert.equal(isRepositoryId("owner/repository"), true);
  for (const value of [
    "repository-only",
    "/invalid",
    "owner/",
    "owner/repo/extra",
    "https://huggingface.co/owner/repo",
    "owner/my repo",
    "../repo",
    "owner/../repo",
  ]) {
    assert.equal(isRepositoryId(value), false, value);
  }
});

const hookSource = readFileSync(
  new URL(
    "../src/features/studio/sections/use-repository-access-validation.ts",
    import.meta.url,
  ),
  "utf8",
);

test("automatic access validation is debounced and stale-safe", () => {
  assert.match(hookSource, /const DEBOUNCE_MS = 650/);
  assert.match(hookSource, /new AbortController\(\)/);
  assert.match(hookSource, /controllerRef\.current\?\.abort\(\)/);
  assert.match(hookSource, /sequenceRef\.current !== sequence/);
  assert.match(hookSource, /if \(!normalizedRepoId\) return/);
  assert.match(hookSource, /normalizedToken/);
});
