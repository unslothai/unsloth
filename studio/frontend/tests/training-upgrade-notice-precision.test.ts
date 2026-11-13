// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// With custom code AND an installable release, the backend answers forces_16bit for the
// 4-bit fallback, so the card must separately disclose that Install switches to 16-bit.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import type { TransformersUpgradeCheck } from "../src/features/transformers-upgrade/types.ts";

register("./helpers/transformers-upgrade-resolver.mjs", import.meta.url);

const { trainingTransformersUpgradeNotice } = await import(
  "../src/features/training/lib/training-transformers-upgrade.ts"
);

const RELEASE = "5.15.0";

const BOTH_ACTIONS: TransformersUpgradeCheck = {
  upgrade: {
    // biome-ignore lint/style/useNamingConvention: API schema
    model_type: "muse_glimmer",
    // biome-ignore lint/style/useNamingConvention: API schema
    pypi_version: RELEASE,
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

test("an offered install that is not already 16-bit discloses that Install switches to 16-bit", () => {
  const notice = trainingTransformersUpgradeNotice(BOTH_ACTIONS, true);
  assert.equal(notice.installVersion, RELEASE);
  assert.equal(notice.fourBitUnavailable, false);
  assert.equal(notice.installSwitchesTo16Bit, true);
});

test("a 16-bit run is not told twice that it will be 16-bit", () => {
  const notice = trainingTransformersUpgradeNotice(
    { ...BOTH_ACTIONS, requiresTrustRemoteCode: false, forces16Bit: true },
    true,
  );
  assert.equal(notice.fourBitUnavailable, true);
  assert.equal(notice.installSwitchesTo16Bit, false);
});

test("a run that never asked for 4-bit has no precision to lose", () => {
  const notice = trainingTransformersUpgradeNotice(BOTH_ACTIONS, false);
  assert.equal(notice.installVersion, RELEASE);
  assert.equal(notice.fourBitUnavailable, false);
  assert.equal(notice.installSwitchesTo16Bit, false);
});

test("a dev-only upgrade has no Install action to warn about", () => {
  const notice = trainingTransformersUpgradeNotice(
    {
      ...BOTH_ACTIONS,
      upgrade: {
        // biome-ignore lint/style/useNamingConvention: API schema
        model_type: "muse_glimmer",
        // biome-ignore lint/style/useNamingConvention: API schema
        pypi_version: RELEASE,
        // biome-ignore lint/style/useNamingConvention: API schema
        supported_in_pypi: false,
        // biome-ignore lint/style/useNamingConvention: API schema
        supported_in_main: true,
      },
    },
    true,
  );
  assert.equal(notice.installVersion, null);
  assert.equal(notice.installSwitchesTo16Bit, false);
});

test("the sidecar already routing this model needs no action-dependent wording", () => {
  const notice = trainingTransformersUpgradeNotice(
    {
      upgrade: null,
      requiresTrustRemoteCode: true,
      latestTierActive: true,
      forces16Bit: true,
      installBreaksExactResume: false,
    },
    true,
  );
  assert.equal(notice.installVersion, null);
  assert.equal(notice.fourBitUnavailable, true);
  assert.equal(notice.installSwitchesTo16Bit, false);
});
