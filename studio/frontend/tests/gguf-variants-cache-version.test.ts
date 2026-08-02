// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  bumpGgufVariantsCacheVersion,
  getGgufVariantsCacheVersion,
} from "../src/features/hub/inventory/gguf-variants-cache-events.ts";

const REPO = "unsloth/Qwen3-8B-GGUF";
const OTHER = "unsloth/Llama-3.1-8B-Instruct-GGUF";

/** What the picker watches: one snapshot over the repos it lists. */
const snapshot = (repoIds: string[]) =>
  repoIds.map((id) => getGgufVariantsCacheVersion(id)).join(",");

test("a per-repo invalidation does not move the global version", () => {
  const before = getGgufVariantsCacheVersion();
  bumpGgufVariantsCacheVersion(REPO);
  // Why watching the global version alone leaves a list stale.
  assert.equal(getGgufVariantsCacheVersion(), before);
});

test("a per-repo invalidation moves that repo's version", () => {
  const before = getGgufVariantsCacheVersion(REPO);
  bumpGgufVariantsCacheVersion(REPO);
  assert.notEqual(getGgufVariantsCacheVersion(REPO), before);
});

test("the aggregated snapshot moves when any listed repo is invalidated", () => {
  const repos = [REPO, OTHER];
  const before = snapshot(repos);
  bumpGgufVariantsCacheVersion(OTHER);
  assert.notEqual(snapshot(repos), before);
});

test("an unrelated repo's invalidation leaves the snapshot alone", () => {
  const repos = [REPO, OTHER];
  const before = snapshot(repos);
  bumpGgufVariantsCacheVersion("unsloth/gemma-3-4b-it-GGUF");
  assert.equal(snapshot(repos), before);
});

test("a global invalidation moves the snapshot too", () => {
  const repos = [REPO, OTHER];
  const before = snapshot(repos);
  bumpGgufVariantsCacheVersion();
  assert.notEqual(snapshot(repos), before);
});
