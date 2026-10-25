// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The cached answer must be invalidated by an install, which provisions a persistent 16-bit sidecar.

import assert from "node:assert/strict";
import test from "node:test";

import {
  hasUpgradeNoticeCache,
  readUpgradeNoticeCache,
  upgradeNoticeCacheKey,
  writeUpgradeNoticeCache,
} from "../src/features/training/lib/training-upgrade-notice-cache.ts";
import type { TransformersUpgradeCheck } from "../src/features/transformers-upgrade/types.ts";

const MODEL = "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit";

const BEFORE_INSTALL: TransformersUpgradeCheck = {
  upgrade: {
    // biome-ignore lint/style/useNamingConvention: API schema
    model_type: "muse_glimmer",
    // biome-ignore lint/style/useNamingConvention: API schema
    pypi_version: "5.15.0",
    // biome-ignore lint/style/useNamingConvention: API schema
    supported_in_pypi: true,
    // biome-ignore lint/style/useNamingConvention: API schema
    supported_in_main: true,
  },
  requiresTrustRemoteCode: true,
  latestTierActive: false,
  forces16Bit: false,
  installBreaksExactResume: false,
};

const AFTER_INSTALL: TransformersUpgradeCheck = {
  upgrade: null,
  requiresTrustRemoteCode: true,
  latestTierActive: true,
  forces16Bit: true,
  installBreaksExactResume: false,
};

test("an answer is reused for the same model, copy and token", () => {
  const key = upgradeNoticeCacheKey(0, MODEL, false, null, "");
  writeUpgradeNoticeCache(0, key, BEFORE_INSTALL);

  assert.equal(upgradeNoticeCacheKey(0, MODEL, false, null, ""), key);
  assert.equal(hasUpgradeNoticeCache(0, key), true);
  assert.equal(readUpgradeNoticeCache(0, key), BEFORE_INSTALL);
});

test("an install retires every answer taken before it", () => {
  const before = upgradeNoticeCacheKey(1, MODEL, false, null, "");
  writeUpgradeNoticeCache(1, before, BEFORE_INSTALL);
  assert.equal(readUpgradeNoticeCache(1, before), BEFORE_INSTALL);

  const after = upgradeNoticeCacheKey(2, MODEL, false, null, "");
  assert.notEqual(after, before);
  assert.equal(
    hasUpgradeNoticeCache(2, after),
    false,
    "a post-install render must re-ask, not repeat the pre-install answer",
  );
  assert.equal(readUpgradeNoticeCache(2, after), null);
});

test("a different copy or token is still a different answer", () => {
  const key = upgradeNoticeCacheKey(3, MODEL, false, null, "");
  writeUpgradeNoticeCache(3, key, BEFORE_INSTALL);

  assert.equal(
    hasUpgradeNoticeCache(
      3,
      upgradeNoticeCacheKey(3, MODEL, true, "/cache/x", ""),
    ),
    false,
  );
  assert.equal(
    hasUpgradeNoticeCache(
      3,
      upgradeNoticeCacheKey(3, MODEL, false, null, "hf_token"),
    ),
    false,
  );
  assert.equal(
    hasUpgradeNoticeCache(
      3,
      upgradeNoticeCacheKey(3, "org/other", false, null, ""),
    ),
    false,
  );
  // A cached row can have a null path; the backend still resolves the pin from cache roots.
  assert.equal(
    hasUpgradeNoticeCache(3, upgradeNoticeCacheKey(3, MODEL, true, null, "")),
    false,
  );
  assert.equal(readUpgradeNoticeCache(3, key), BEFORE_INSTALL);
});

test("a check still in flight across the install cannot rewind the cache", () => {
  // A stale pre-install check can resolve after the post-install one; its write must be dropped
  // or nothing re-asks and the notice vanishes.
  const stale = upgradeNoticeCacheKey(4, MODEL, false, null, "");
  const fresh = upgradeNoticeCacheKey(5, MODEL, false, null, "");
  writeUpgradeNoticeCache(5, fresh, AFTER_INSTALL);

  writeUpgradeNoticeCache(4, stale, BEFORE_INSTALL);

  assert.equal(hasUpgradeNoticeCache(5, fresh), true);
  assert.equal(readUpgradeNoticeCache(5, fresh), AFTER_INSTALL);
  assert.equal(hasUpgradeNoticeCache(4, stale), false);
  assert.equal(readUpgradeNoticeCache(4, stale), null);
});
