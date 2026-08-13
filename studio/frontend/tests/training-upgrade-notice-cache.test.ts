// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Configure preview asks the upgrade check once per model and caches the answer,
// because the hook behind it runs on every render. That cache used to outlive the thing
// it describes: consenting to the install on Start provisions a persistent sidecar, and
// coming back to Configure in the same session kept reading the pre-install answer -- so
// the card went on offering a release that was already installed, and went on promising
// "QLoRA - 4-bit" for a run the new overlay loads in 16-bit, which is the threefold VRAM
// understatement this preview exists to prevent.

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

// What the check says before the install: a release to offer, and 4-bit still available
// because the model's own code can load it on the current transformers.
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

  // The consent flow installed the sidecar, so the store's generation moved on.
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
  // A known-cached row can have a null path, and the backend still resolves the pin from
  // the cache roots, so the flag alone is a different question about the same model.
  assert.equal(
    hasUpgradeNoticeCache(3, upgradeNoticeCacheKey(3, MODEL, true, null, "")),
    false,
  );
  assert.equal(readUpgradeNoticeCache(3, key), BEFORE_INSTALL);
});
